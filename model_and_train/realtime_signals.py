"""Causal, latest-date ETF signal snapshot service."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .panel_inference import aggregate_panel_predictions


class FixedHorizonExecutionPlanner:
    """Map daily ranking snapshots to fixed-horizon portfolio orders.

    The planner mirrors the research backtest rule: a selected batch is opened
    on the next trading day and new ranking signals are ignored until the
    batch's planned exit date.
    """

    def __init__(self, trading_dates, hold_days: int = 10, top_k: int = 2):
        dates = list(trading_dates)
        if not dates:
            raise ValueError("trading_dates 不能为空")
        if hold_days <= 0 or top_k <= 0:
            raise ValueError("hold_days 和 top_k 必须大于 0")
        if len(set(dates)) != len(dates):
            raise ValueError("trading_dates 不能重复")
        self.trading_dates = dates
        self.hold_days = int(hold_days)
        self.top_k = int(top_k)
        self.active_until = None

    def plan(self, snapshot: pd.DataFrame, prices: pd.DataFrame | None = None) -> dict:
        required = {"trade_date", "symbol", "selected"}
        missing = required - set(snapshot.columns)
        if missing:
            raise ValueError(f"snapshot 缺少字段: {sorted(missing)}")
        if snapshot.empty:
            raise ValueError("snapshot 不能为空")
        dates = snapshot["trade_date"].dropna().unique().tolist()
        if len(dates) != 1:
            raise ValueError("snapshot 必须只包含一个 trade_date")
        signal_date = dates[0]
        if signal_date not in self.trading_dates:
            raise ValueError("trade_date 不在交易日历")

        if self.active_until is not None and signal_date <= self.active_until:
            selected_count = int(snapshot["selected"].astype(bool).sum())
            return {
                "action": "skip_overlap",
                "signal_date": signal_date,
                "execute_date": None,
                "active_until": self.active_until,
                "orders": [],
                "skipped_signal_count": selected_count,
            }

        selected = snapshot[snapshot["selected"].astype(bool)].copy()
        if selected.empty:
            self.active_until = None
            return {
                "action": "hold",
                "signal_date": signal_date,
                "execute_date": None,
                "active_until": None,
                "orders": [],
                "skipped_signal_count": 0,
            }
        if len(selected) > self.top_k:
            raise ValueError("选中标的数量不能超过 top_k")

        signal_index = self.trading_dates.index(signal_date)
        execute_index = signal_index + 1
        exit_index = signal_index + self.hold_days
        if exit_index >= len(self.trading_dates):
            latest_valid_index = len(self.trading_dates) - 1 - self.hold_days
            latest_valid_signal = (
                self.trading_dates[latest_valid_index]
                if latest_valid_index >= 0
                else "不可用"
            )
            raise ValueError(
                f"signal_date 后没有完整持有窗口；数据最后交易日={self.trading_dates[-1]}；"
                f"建议最晚信号日={latest_valid_signal}"
            )
        execute_date = (
            self.trading_dates[execute_index]
            if execute_index < len(self.trading_dates)
            else None
        )
        if execute_date is None:
            raise ValueError("signal_date 后没有可用的 T+1 交易日")
        active_until = self.trading_dates[exit_index]
        self.active_until = active_until
        default_weight = 1.0 / len(selected)
        orders = []
        for row in selected.itertuples(index=False):
            weight = getattr(row, "weight", default_weight)
            if pd.isna(weight):
                weight = default_weight
            order = {
                "symbol": str(row.symbol),
                "weight": float(weight),
                "signal_date": signal_date,
                "execute_date": execute_date,
                "planned_exit_date": active_until,
                "status": "executable",
                "execution_price": None,
                "reject_reason": None,
            }
            if prices is not None:
                order.update(self._check_executability(prices, str(row.symbol), execute_date))
            orders.append(order)
        if prices is not None and not any(order["status"] == "executable" for order in orders):
            self.active_until = None
            return {
                "action": "no_executable_orders",
                "signal_date": signal_date,
                "execute_date": execute_date,
                "active_until": None,
                "orders": orders,
                "skipped_signal_count": 0,
            }
        return {
            "action": "open",
            "signal_date": signal_date,
            "execute_date": execute_date,
            "active_until": active_until,
            "orders": orders,
            "skipped_signal_count": 0,
        }

    @staticmethod
    def _check_executability(prices: pd.DataFrame, symbol: str, execute_date: Any) -> dict:
        required = {"trade_date", "symbol", "open"}
        missing = required - set(prices.columns)
        if missing:
            raise ValueError(f"prices 缺少字段: {sorted(missing)}")
        rows = prices[(prices["symbol"].astype(str) == symbol) & (prices["trade_date"] == execute_date)]
        if rows.empty:
            return {"status": "rejected", "execution_price": None, "reject_reason": "t_plus_one_missing"}
        row = rows.iloc[0]
        is_suspend = int(row.get("is_suspend", 0))
        if is_suspend != 0:
            return {"status": "rejected", "execution_price": None, "reject_reason": "t_plus_one_suspended"}
        price = pd.to_numeric(pd.Series([row["open"]]), errors="coerce").iloc[0]
        if pd.isna(price) or float(price) <= 0:
            return {"status": "rejected", "execution_price": None, "reject_reason": "t_plus_one_open_invalid"}
        return {"status": "executable", "execution_price": float(price), "reject_reason": None}


class ExecutionPlanStore:
    """Persist execution plans and append broker fill results."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, plan: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(plan)
        signals = payload.get("signals")
        if isinstance(signals, pd.DataFrame):
            payload["signals"] = signals.to_dict("records")
        self.path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")

    def load(self) -> dict:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload.get("signals"), list):
            payload["signals"] = pd.DataFrame(payload["signals"])
        return payload

    def record_fills(self, plan: dict, fills: list[dict]) -> dict:
        fills_by_symbol = {str(fill["symbol"]): fill for fill in fills if "symbol" in fill}
        updated = dict(plan)
        updated["orders"] = []
        for order in plan.get("orders", []):
            item = dict(order)
            fill = fills_by_symbol.get(str(item.get("symbol")))
            if fill is not None:
                history = [dict(entry) for entry in item.get("fill_history", [])]
                history.append(dict(fill))
                item["fill_history"] = history
                requested = item.get("requested_quantity")
                filled_quantity = fill.get("filled_quantity")
                if not bool(fill.get("filled")):
                    item.update({key: value for key, value in fill.items() if key != "filled_quantity"})
                    item["status"] = "fill_rejected"
                elif requested is not None and filled_quantity is not None:
                    requested = float(requested)
                    current_quantity = float(filled_quantity)
                    previous_quantity = float(item.get("filled_quantity", 0.0) or 0.0)
                    total_quantity = previous_quantity + current_quantity
                    previous_price = item.get("fill_price")
                    current_price = fill.get("fill_price")
                    if current_price is not None and total_quantity > 0:
                        if previous_price is not None and previous_quantity > 0:
                            item["fill_price"] = (
                                previous_price * previous_quantity
                                + float(current_price) * current_quantity
                            ) / total_quantity
                        else:
                            item["fill_price"] = float(current_price)
                    item["filled_quantity"] = total_quantity
                    item["remaining_quantity"] = max(requested - total_quantity, 0.0)
                    item["status"] = (
                        "filled" if total_quantity >= requested else "partially_filled"
                    )
                else:
                    item.update(fill)
                    item["status"] = "filled"
            updated["orders"].append(item)
        self.save(updated)
        return updated

    def mark_submitted(self, plan: dict) -> dict:
        """Mark executable orders submitted without allowing duplicate submits."""
        updated = dict(plan)
        updated["orders"] = []
        submitted_count = 0
        duplicate_count = 0
        signal_date = plan.get("signal_date")
        execute_date = plan.get("execute_date")
        for order in plan.get("orders", []):
            item = dict(order)
            item.setdefault(
                "order_id", f"{signal_date}-{execute_date}-{item.get('symbol')}"
            )
            status = str(item.get("status", "executable"))
            if status in {"executable", "ready"}:
                item["status"] = "submitted"
                submitted_count += 1
            else:
                duplicate_count += 1
            updated["orders"].append(item)
        updated["submitted_count"] = submitted_count
        updated["duplicate_count"] = duplicate_count
        self.save(updated)
        return updated

    def export_manual_checklist(self, plan: dict, path: str | Path) -> Path:
        """Export a human-reviewable order checklist without submitting orders."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for order in plan.get("orders", []):
            status = str(order.get("status", ""))
            if status in {"executable", "ready"}:
                manual_action = "待人工确认"
            elif status == "filled":
                manual_action = "已记录成交"
            elif status == "partially_filled":
                manual_action = "部分成交，待人工处理"
            elif status == "fill_rejected":
                manual_action = "成交拒绝，待人工处理"
            elif status == "rejected":
                manual_action = "不可成交"
            else:
                manual_action = "复核状态"
            rows.append({
                "order_id": order.get("order_id"),
                "symbol": order.get("symbol"),
                "signal_date": order.get("signal_date", plan.get("signal_date")),
                "execute_date": order.get("execute_date", plan.get("execute_date")),
                "planned_exit_date": order.get("planned_exit_date", plan.get("active_until")),
                "weight": order.get("weight"),
                "status": status,
                "manual_action": manual_action,
                "execution_price": order.get("execution_price"),
                "fill_price": order.get("fill_price"),
                "requested_quantity": order.get("requested_quantity"),
                "filled_quantity": order.get("filled_quantity"),
                "remaining_quantity": order.get("remaining_quantity"),
                "reject_reason": order.get("reject_reason"),
                "manual_confirmed": order.get("manual_confirmed"),
                "manual_abandoned": order.get("manual_abandoned"),
                "actual_exit_date": order.get("actual_exit_date"),
                "actual_exit_price": order.get("actual_exit_price"),
                "manual_notes": order.get("manual_notes"),
            })
        pd.DataFrame(rows).to_csv(target, index=False, encoding="utf-8-sig")
        return target


class RealtimeSignalService:
    """Generate a reproducible signal snapshot from already-fetched predictions."""

    def __init__(
        self,
        top_k: int = 2,
        min_pred_return: float | None = None,
        min_event_probability: float | None = None,
        score_weights: dict[str, float] | None = None,
    ) -> None:
        self.top_k = int(top_k)
        self.min_pred_return = min_pred_return
        self.min_event_probability = min_event_probability
        self.score_weights = score_weights

    def generate(
        self,
        predictions_by_window: Mapping[int, tuple[pd.DataFrame, Mapping[str, Any]]],
        as_of_date: Any,
    ) -> dict:
        """Return only the latest prediction date not later than ``as_of_date``."""
        filtered = {}
        for look_back, (meta, prediction) in predictions_by_window.items():
            if "trade_date" not in meta.columns:
                raise ValueError(f"窗口 {look_back}: meta 缺少 trade_date")
            item = meta[meta["trade_date"] <= as_of_date].copy()
            if len(item):
                indices = item.index.to_numpy()
                filtered_prediction = {
                    key: value[indices] for key, value in prediction.items()
                }
                filtered[int(look_back)] = (item.reset_index(drop=True), filtered_prediction)
        if not filtered:
            raise ValueError("没有不晚于 as_of_date 的可用预测")
        signals = aggregate_panel_predictions(
            filtered,
            top_k=self.top_k,
            min_pred_return=self.min_pred_return,
            min_event_probability=self.min_event_probability,
            score_weights=self.score_weights,
        )
        latest_date = signals["trade_date"].max()
        snapshot = signals[signals["trade_date"] == latest_date].copy()
        if (snapshot["trade_date"] > as_of_date).any():
            raise RuntimeError("实时信号包含未来日期")
        return {
            "as_of_date": as_of_date,
            "trade_date": latest_date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "top_k": self.top_k,
            "signals": snapshot.reset_index(drop=True),
        }

    def generate_execution_plan(
        self,
        predictions_by_window: Mapping[int, tuple[pd.DataFrame, Mapping[str, Any]]],
        as_of_date: Any,
        planner: FixedHorizonExecutionPlanner,
        prices: pd.DataFrame | None = None,
    ) -> dict:
        """Generate a ranked snapshot and apply the shared execution policy."""
        snapshot = self.generate(predictions_by_window, as_of_date)
        plan = planner.plan(snapshot["signals"], prices=prices)
        plan["generated_at"] = snapshot["generated_at"]
        plan["trade_date"] = snapshot["trade_date"]
        plan["signals"] = snapshot["signals"]
        return plan
