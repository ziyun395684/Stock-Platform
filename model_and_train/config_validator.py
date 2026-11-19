"""Task 4.3 配置文件校验器 ConfigValidator.

对应 Todo Task4.3：`config.yaml` 模板 + 新增 `ConfigValidator` 校验器；
非法配置启动直接报错（错误信息清晰定位字段）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """配置校验失败（携带字段路径与原因）。"""


def _fmt_path(path: List[str]) -> str:
    return ".".join(path) if path else "<root>"


class ConfigValidator:
    """YAML 配置校验器（Task4.3）。

    支持：
        * 必填字段缺失检测；
        * 类型检查（int/float/bool/str/list）；
        * 数值范围检查（min/max）；
        * 枚举取值检查（choices）；
        * 嵌套 dict 递归校验。
    """

    #: 必填字段（缺失即报错）。
    REQUIRED: tuple = (
        "data", "model", "training", "ensemble", "report", "paths",
    )

    #: 字段规则：path -> dict(required/type/min/max/choices/default)。
    RULES: Dict[str, Dict[str, Any]] = {
        # ---- ETF universe ----
        "universe.symbols": {"type": list, "min_items": 1, "default": ["510300.SH"]},
        "universe.benchmark": {"type": str, "default": "510300.SH"},
        "universe.top_k": {"type": int, "min": 1, "max": 100, "default": 2},
        "universe.correlation_threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.85},
        "universe.allow_short_history": {"type": bool, "default": False},
        "universe.date_alignment": {"type": str, "choices": ["common", "union"], "default": "common"},
        # ---- data ----
        "data.source": {"required": True, "type": str, "choices": ["csv", "akshare", "tushare"]},
        "data.code": {"required": True, "type": str},
        "data.start": {"required": True, "type": str},
        "data.end": {"required": True, "type": str},
        "data.adjust_type": {"required": True, "type": str, "choices": ["qfq", "hfq", "none"]},
        "data.min_years": {"type": (int, float), "min": 1, "max": 20, "default": 5},
        "data.extended_panel_features": {"type": bool, "default": False},
        "data.panel_feature_columns": {"type": (list, type(None)), "default": None},
        # ---- model ----
        "model.look_backs": {"required": True, "type": list, "min_items": 1},
        "model.lstm_units": {"type": int, "min": 8, "max": 256, "default": 64},
        "model.transformer_layers": {"type": int, "min": 1, "max": 4, "default": 2},
        "model.num_heads": {"type": int, "min": 1, "max": 16, "default": 4},
        "model.dropout": {"type": float, "min": 0.0, "max": 0.9, "default": 0.3},
        "model.max_relative_distance": {"type": int, "min": 1, "max": 200, "default": 120},
        "model.regression_loss_weight": {"type": float, "min": 0.0, "max": 100.0, "default": 1.0},
        "model.classification_loss_weight": {"type": float, "min": 0.0, "max": 100.0, "default": 1.0},
        "model.return_clip": {"type": float, "min": 0.0, "max": 1.0, "default": 0.2},
        "model.return_target_scaling": {"type": bool, "default": False},
        "model.ranking_target": {"type": str, "choices": ["raw", "cross_sectional", "factor_residual"], "default": "raw"},
        "model.ranking_loss_weight": {"type": float, "min": 0.0, "max": 100.0, "default": 0.0},
        "model.dispersion_loss_weight": {"type": float, "min": 0.0, "max": 100.0, "default": 0.0},
        "model.dispersion_target_std": {"type": float, "min": 0.0, "max": 1.0, "default": 0.0},
        "model.batch_mode": {"type": str, "choices": ["auto", "random", "date_grouped"], "default": "auto"},
        "model.validation_monitor": {"type": str, "choices": ["loss", "rank_ic"], "default": "rank_ic"},
        "model.use_symbol_one_hot": {"type": bool, "default": False},
        "model.factor_guidance_weight": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
        "model.factor_guidance_schedule": {"type": (list, type(None)), "default": None},
        "model.factor_guidance_epochs_per_stage": {"type": int, "min": 1, "max": 1000, "default": 1},
        "model.indicator_auxiliary_weight": {"type": float, "min": 0.0, "max": 100.0, "default": 0.0},
        "model.indicator_target_columns": {"type": (list, type(None)), "default": None},
        # ---- training ----
        "training.learning_rate": {"type": float, "min": 1e-6, "max": 1.0, "default": 1e-3},
        "training.batch_size": {"type": int, "min": 1, "max": 1024, "default": 32},
        "training.epochs": {"type": int, "min": 1, "max": 1000, "default": 100},
        "training.early_stop_patience": {"type": int, "min": 1, "max": 200, "default": 15},
        "training.clipnorm": {"type": float, "min": 0.01, "max": 100.0, "default": 1.0},
        "training.max_class_weight": {"type": float, "min": 1.0, "max": 100.0, "default": 5.0},
        "training.min_positive_count": {"type": int, "min": 0, "default": 50},
        "training.min_positive_ratio": {"type": float, "min": 0.0, "max": 1.0, "default": 0.05},
        "training.min_sample_count": {"type": int, "min": 1, "default": 50},
        "training.parallel_training": {"type": bool, "default": False},
        "training.random_seed": {"type": (int, type(None)), "default": None},
        # ---- data split ----
        "split.split_ratios": {"type": dict, "default": {"train": 0.70, "val": 0.15, "test": 0.15}},
        "split.split_mode": {"type": str, "choices": ["base", "enhance"], "default": "base"},
        "split.enable_lowvol_weight": {"type": bool, "default": True},
        "split.embargo_days": {"type": int, "min": 0, "max": 30, "default": 3},
        # ---- rolling OOS ----
        "rolling.train_days": {"type": int, "min": 10, "default": 1000},
        "rolling.val_days": {"type": int, "min": 1, "default": 250},
        "rolling.test_days": {"type": int, "min": 1, "default": 250},
        "rolling.step_days": {"type": int, "min": 1, "default": 250},
        # ---- labels ----
        "labels.label_mode": {"type": str, "choices": ["tolerant", "strict"], "default": "tolerant"},
        "labels.label_threshold": {"type": float, "min": 0.0, "max": 0.5, "default": 0.05},
        # ---- ensemble ----
        "ensemble.mode": {"type": str, "choices": ["baseline", "enhanced"], "default": "baseline"},
        "ensemble.infer_threshold": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
        # ---- backtest ----
        "backtest.cost_rate": {"type": float, "min": 0.0, "max": 0.1, "default": 0.003},
        "backtest.slippage_rate": {"type": float, "min": 0.0, "max": 0.99, "default": 0.0},
        "backtest.initial_capital": {"type": float, "min": 0.0, "default": 1.0},
        "backtest.hold_days": {"type": int, "min": 1, "max": 252, "default": 3},
        "backtest.min_signal_interval": {"type": int, "min": 0, "default": 0},
        "backtest.exit_mode": {"type": str, "choices": ["fixed", "dynamic", "signal"], "default": "fixed"},
        "backtest.dynamic_enabled": {"type": bool, "default": False},
        "backtest.stop_loss": {"type": (float, type(None)), "min": 0.0, "max": 1.0, "default": None},
        "backtest.trend_exit_days": {"type": (int, type(None)), "min": 1, "default": None},
        "backtest.rank_exit_threshold": {"type": (int, type(None)), "min": 1, "default": None},
        "backtest.max_hold_days": {"type": int, "min": 1, "max": 252, "default": 3},
        # ---- factor/model hybrid ranking ----
        "ranking.factor_weight": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
        "ranking.model_weight": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
        "ranking.weight_source": {"type": str, "choices": ["config", "validation"], "default": "config"},
        "ranking.min_pred_return": {"type": (float, type(None)), "default": None},
        "ranking.min_event_probability": {"type": (float, type(None)), "min": 0.0, "max": 1.0, "default": None},
        # ---- report ----
        "report.output_csv": {"type": str, "default": "output/report.csv"},
        "report.output_json": {"type": str, "default": "output/report.json"},
        # ---- paths ----
        "paths.data_dir": {"type": str, "default": "quotes"},
        "paths.model_dir": {"type": str, "default": "model_and_train/checkpoints"},
        "paths.output_dir": {"type": str, "default": "output"},
    }

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config: Dict = config or {}

    # ------------------------------------------------------------------
    def _get(self, path: List[str]) -> Any:
        cur: Any = self.config
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    @staticmethod
    def _denormalize(flat: Dict[str, Any]) -> Dict[str, Any]:
        """把扁平点路径 dict（{"data.source": x}）还原为嵌套 dict。"""
        out: Dict[str, Any] = {}
        for k, v in flat.items():
            parts = k.split(".")
            cur = out
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = v
        return out

    def validate(self, config: Optional[Dict] = None) -> Dict:
        """校验配置；返回规范化后的配置（填充默认值）。

        输入支持两种结构：
            * 嵌套 dict（含顶层 data/model/... 节）；
            * 扁平点路径 dict（如 {"data.source": "csv"}，CLI 覆盖后的形态）。

        Raises:
            ConfigValidationError: 非法配置（缺失/类型/范围/取值）。
        """
        if config is not None:
            self.config = config
        cfg = self.config
        if not isinstance(cfg, dict):
            raise ConfigValidationError(f"配置必须是 dict，收到 {type(cfg)!r}")

        # 扁平点路径输入 -> 还原嵌套再校验。
        has_top = any(top in cfg for top in self.REQUIRED)
        has_dotted = any("." in k for k in cfg)
        if not has_top and has_dotted:
            self.config = self._denormalize(cfg)
            cfg = self.config

        # 1) 必填顶层字段。
        for top in self.REQUIRED:
            if top not in cfg:
                raise ConfigValidationError(f"缺少必填配置节: {top!r}")

        # 2) 逐字段校验（含嵌套）。
        normalized: Dict[str, Any] = {}
        for dotted, rule in self.RULES.items():
            path = dotted.split(".")
            value = self._get(path)
            normalized[dotted] = self._check(dotted, value, rule)

        schedule = normalized.get("model.factor_guidance_schedule")
        if schedule is not None:
            try:
                values = [float(value) for value in schedule]
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError(
                    "model.factor_guidance_schedule 必须是数值列表"
                ) from exc
            if len(values) < 2 or any(
                left < right for left, right in zip(values, values[1:])
            ):
                raise ConfigValidationError(
                    "model.factor_guidance_schedule 必须至少两项且递减"
                )
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ConfigValidationError(
                    "model.factor_guidance_schedule 权重必须位于 [0, 1]"
                )

        indicator_weight = float(normalized["model.indicator_auxiliary_weight"])
        indicator_columns = normalized["model.indicator_target_columns"]
        if indicator_weight > 0.0:
            if not indicator_columns:
                raise ConfigValidationError(
                    "启用指标辅助任务时 model.indicator_target_columns 不能为空"
                )
            if any(not isinstance(column, str) or not column.strip() for column in indicator_columns):
                raise ConfigValidationError(
                    "model.indicator_target_columns 必须是非空字符串列表"
                )

        factor_weight = float(normalized["ranking.factor_weight"])
        model_weight = float(normalized["ranking.model_weight"])
        if factor_weight + model_weight <= 0.0:
            raise ConfigValidationError(
                "ranking.factor_weight 与 ranking.model_weight 不能同时为 0"
            )
        return normalized

    def _check(self, dotted: str, value: Any, rule: Dict[str, Any]) -> Any:
        """单字段校验（缺失->默认值；类型/范围/枚举校验）。"""
        required = rule.get("required", False)
        if value is None:
            if required:
                raise ConfigValidationError(f"缺少必填字段: {dotted}")
            return rule.get("default")

        # 类型检查。
        expected = rule.get("type")
        if expected is not None:
            ok = isinstance(value, expected)
            # bool 是 int 子类，单独处理避免 bool 通过 int 检查。
            if expected is bool and isinstance(value, bool):
                ok = True
            elif expected is int and isinstance(value, bool):
                ok = False
            if not ok:
                raise ConfigValidationError(
                    f"字段 {dotted}: 期望类型 {expected}，收到 {type(value).__name__}"
                )

        # 数值范围。
        vmin, vmax = rule.get("min"), rule.get("max")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if vmin is not None and value < vmin:
                raise ConfigValidationError(f"字段 {dotted}: {value} < 下限 {vmin}")
            if vmax is not None and value > vmax:
                raise ConfigValidationError(f"字段 {dotted}: {value} > 上限 {vmax}")

        # 枚举取值。
        choices = rule.get("choices")
        if choices is not None and value not in choices:
            raise ConfigValidationError(
                f"字段 {dotted}: 取值 {value!r} 不在 {choices} 中"
            )

        # list 最小项数。
        min_items = rule.get("min_items")
        if min_items is not None and isinstance(value, (list, tuple)) and len(value) < min_items:
            raise ConfigValidationError(
                f"字段 {dotted}: 至少 {min_items} 项，收到 {len(value)}"
            )
        return value

    # ------------------------------------------------------------------
    def load_yaml(self, path: str) -> Dict:
        """读取 YAML 并校验。"""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return self.validate(cfg)
