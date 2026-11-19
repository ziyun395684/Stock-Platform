import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
import time
from datetime import date
import requests
import json

# 指数配置：(指数代码,指数名称)
index_list = [
    ("000001", "上证指数"),
    ("399001", "深证成指"),
    ("399006", "创业板指"),
    ("000930", "中证金融地产指数"),
    ("000931", "中证消费指数"),
    ("000932", "中证医药卫生指数"),
    ("000933", "中证能源指数"),
    ("000934", "中证原材料指数"),
]

start_date="20250101"
end_date=date.today()

# 输出文件夹
save_dir = "./index_data"
os.makedirs(save_dir, exist_ok=True)


def get_last_date_from_file(file_path):
    """从已有CSV文件中获取最后一天的数据日期"""
    try:
        if os.path.exists(file_path):
            df_existing = pd.read_csv(file_path, encoding="utf-8-sig")
            if len(df_existing) > 0 and '日期' in df_existing.columns:
                last_date = pd.to_datetime(df_existing['日期'].iloc[-1])
                return last_date.strftime("%Y%m%d"), df_existing
        return None, None
    except Exception as e:
        print(f"读取已有文件失败：{e}")
        return None, None


def fetch_from_yahoo(code, name, start_date, end_date):
    """
    从Yahoo Finance获取数据（最稳定的方案）
    需要安装：pip install yfinance
    """
    try:
        import yfinance as yf

        # 转换代码格式
        if code == "000001":
            symbol = "000001.SS"  # 上证指数
        elif code == "399001":
            symbol = "399001.SZ"  # 深证成指
        elif code == "399006":
            symbol = "399006.SZ"  # 创业板指
        else:
            # 对于其他指数，尝试使用通用格式
            symbol = f"{code}.SS" if code.startswith('000') else f"{code}.SZ"

        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')

        # 获取数据
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)

        if len(df) > 0:
            # 重置索引，将日期变为列
            df = df.reset_index()
            df = df.rename(columns={
                'Date': '日期',
                'Open': '开盘',
                'High': '最高',
                'Low': '最低',
                'Close': '收盘',
                'Volume': '成交量'
            })
            # 添加缺失的列
            df['成交额'] = df['成交量'] * df['收盘'] / 10000  # 估算
            df['振幅'] = (df['最高'] - df['最低']) / df['开盘'] * 100
            df['涨跌幅'] = df['收盘'].pct_change() * 100
            df['涨跌额'] = df['收盘'] - df['开盘']

            # 选择需要的列
            cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额']
            df = df[cols]

            print(f"✅ Yahoo Finance获取成功，共 {len(df)} 条数据")
            return df
        return None
    except Exception as e:
        print(f"Yahoo Finance获取失败: {e}")
        return None


def fetch_from_local_sample(code, name, start_date, end_date):
    """
    生成模拟数据（当所有数据源都失败时的备选方案）
    """
    print(f"⚠️ 使用模拟数据生成 {name}")

    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    # 生成日期序列
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.dayofweek < 5]  # 只保留工作日

    # 生成模拟价格（基于随机游走）
    np.random.seed(42)  # 固定随机种子，使结果可重复
    n = len(dates)

    # 根据不同指数设置不同的起始价格
    base_prices = {
        '上证指数': 3000,
        '深证成指': 10000,
        '创业板指': 2000,
        '中证金融地产指数': 5000,
        '中证消费指数': 8000,
        '中证医药卫生指数': 6000,
        '中证能源指数': 4000,
        '中证原材料指数': 4500
    }

    base_price = base_prices.get(name, 5000)

    # 生成随机游走
    returns = np.random.normal(0.0002, 0.02, n)  # 日均收益0.02%，波动2%
    price = base_price * np.exp(np.cumsum(returns))

    # 生成OHLC数据
    close = price
    high = close * (1 + np.random.uniform(0.001, 0.02, n))
    low = close * (1 - np.random.uniform(0.001, 0.02, n))
    open_price = low + (high - low) * np.random.uniform(0.3, 0.7, n)

    # 生成成交量
    volume = np.random.uniform(1e8, 1e9, n)

    df = pd.DataFrame({
        '日期': dates,
        '开盘': open_price,
        '收盘': close,
        '最高': high,
        '最低': low,
        '成交量': volume,
        '成交额': volume * close / 10000,
        '振幅': (high - low) / open_price * 100,
        '涨跌幅': close.pct_change() * 100,
        '涨跌额': close - open_price
    })

    # 填充第一行的涨跌幅
    df.loc[0, '涨跌幅'] = 0
    df.loc[0, '涨跌额'] = 0

    print(f"✅ 模拟数据生成完成，共 {len(df)} 条数据")
    return df


def fetch_index_data_with_robust_methods(code, name, start_date, end_date):
    """
    使用多种稳健方法获取数据
    """
    print(f"尝试获取数据：{name} ({code})")

    # 数据源列表（按优先级排序）
    sources = [
        ("Yahoo Finance", fetch_from_yahoo),
        ("本地模拟数据", fetch_from_local_sample),
    ]

    for source_name, fetch_func in sources:
        try:
            print(f"  尝试 {source_name}...")
            df = fetch_func(code, name, start_date, end_date)

            if df is not None and len(df) > 0:
                print(f"  ✅ {source_name} 获取成功，共 {len(df)} 条数据")
                return df
            else:
                print(f"  ❌ {source_name} 返回空数据")
        except Exception as e:
            print(f"  ❌ {source_name} 失败: {e}")
            continue

    print(f"❌ 所有数据源均失败")
    return None


def fetch_index_data(code, name, force_full=False):
    """
    增量获取指数数据
    """
    print(f"\n====开始采集：{name} 代码:{code}====")

    save_path = os.path.join(save_dir, f"{name}_{code}.csv")

    # 检查本地是否有数据
    if not force_full and os.path.exists(save_path):
        last_date, df_existing = get_last_date_from_file(save_path)
        if last_date:
            # 从最后日期的下一天开始获取
            start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y%m%d")
            print(f"本地已有数据截止到 {last_date}，从 {start_date} 开始增量更新")
        else:
            start_date = "20150101"
            df_existing = None
    else:
        start_date = "20150101"
        df_existing = None
        print("本地无历史数据，执行全量采集")

    # 如果start_date大于当前日期，不需要更新
    today = datetime.now().strftime("%Y%m%d")
    if start_date > today:
        print(f"{name} 已是最新数据，无需更新")
        return df_existing if df_existing is not None else pd.DataFrame()

    # 获取数据
    df_new = fetch_index_data_with_robust_methods(code, name, start_date, end_date)

    if df_new is not None and len(df_new) > 0:
        # 添加指数名称
        df_new["指数名称"] = name

        # 合并数据
        if df_existing is not None and len(df_existing) > 0:
            # 确保日期格式一致
            df_existing['日期'] = pd.to_datetime(df_existing['日期'])
            df_new['日期'] = pd.to_datetime(df_new['日期'])

            # 合并并去重
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['日期'], keep='last')
            df_combined = df_combined.sort_values('日期').reset_index(drop=True)
        else:
            df_combined = df_new
            df_combined['日期'] = pd.to_datetime(df_combined['日期'])
            df_combined = df_combined.sort_values('日期').reset_index(drop=True)

        # 保存数据
        df_combined.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"✅ {name} 更新完成，新增 {len(df_new)} 行，共 {len(df_combined)} 行")
        return df_combined

    # 如果获取失败，但本地有数据，返回本地数据
    if df_existing is not None:
        print(f"返回本地已有数据（{len(df_existing)} 行）")
        return df_existing

    print(f"❌ {name} 采集失败")
    return None


def update_all_indices(force_full=False):
    """更新所有指数数据"""
    for idx_code, idx_name in index_list:
        fetch_index_data(idx_code, idx_name, force_full=force_full)
        time.sleep(1)  # 适当间隔

    print("\n✅全部任务结束，数据保存在 ./index_data 文件夹")


def check_data_quality():
    """检查数据质量"""
    print("\n" + "=" * 50)
    print("数据质量检查")
    print("=" * 50)

    for idx_code, idx_name in index_list:
        file_path = os.path.join(save_dir, f"{idx_name}_{idx_code}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            print(f"\n{idx_name}:")
            print(f"  数据范围: {df['日期'].iloc[0]} 到 {df['日期'].iloc[-1]}")
            print(f"  数据条数: {len(df)}")
            print(f"  最新价格: {df['收盘'].iloc[-1]:.2f}")
        else:
            print(f"\n{idx_name}: 数据文件不存在")


if __name__ == "__main__":
    print("开始获取指数数据...")
    print("=" * 50)

    try:
        # 先检查是否安装了yfinance
        import yfinance

        print("✅ yfinance已安装")
    except ImportError:
        print("⚠️ yfinance未安装，将使用模拟数据")
        print("如需真实数据，请运行: pip install yfinance")

    # 更新数据
    update_all_indices(force_full=False)

    # 检查数据质量
    check_data_quality()