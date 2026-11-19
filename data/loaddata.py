import akshare as ak
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# 指数配置：(指数代码,指数名称)
index_list = [
    ("000001", "上证指数"),
    ("399001", "深证成指"),
    ("399006", "创业板指"),
    # 行业指数，可以自行增减代码
    ("000930", "中证金融地产指数"),
    ("000931", "中证消费指数"),
    ("000932", "中证医药卫生指数"),
    ("000933", "中证能源指数"),
    ("000934", "中证原材料指数"),
]

# 时间范围 YYYYMMDD
START_DATE = "20150101"
END_DATE = "20260822"

# 输出文件夹
save_dir = "./index_data"
os.makedirs(save_dir, exist_ok=True)


def get_last_date_from_file(file_path):
    """从已有CSV文件中获取最后一天的数据日期"""
    try:
        df_existing = pd.read_csv(file_path, encoding="utf-8-sig")
        if len(df_existing) > 0:
            # 假设日期列名为'日期'
            last_date = pd.to_datetime(df_existing['日期'].iloc[-1])
            return last_date.strftime("%Y%m%d"), df_existing
        return None, None
    except Exception as e:
        print(f"读取已有文件失败：{e}")
        return None, None


def fetch_index_data(code, name, force_full=False):
    """
    增量获取指数数据
    force_full: 强制全量更新（覆盖）
    """
    print(f"====开始采集：{name} 代码:{code}====")

    save_path = os.path.join(save_dir, f"{name}_{code}.csv")

    # 检查本地是否有数据
    if not force_full and os.path.exists(save_path):
        last_date, df_existing = get_last_date_from_file(save_path)
        if last_date:
            # 从最后日期的下一天开始获取
            start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y%m%d")
            print(f"本地已有数据截止到 {last_date}，从 {start_date} 开始增量更新")
        else:
            start_date = START_DATE
            df_existing = None
    else:
        start_date = START_DATE
        df_existing = None
        print("本地无历史数据，执行全量采集")

    # 如果start_date大于当前日期，不需要更新
    today = datetime.now().strftime("%Y%m%d")
    if start_date > today:
        print(f"{name} 已是最新数据，无需更新")
        return df_existing if df_existing is not None else pd.DataFrame()

    try:
        # 获取新数据
        df_new = ak.index_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=END_DATE
        )

        if df_new is None or len(df_new) == 0:
            print(f"{name} 没有新增数据")
            return df_existing if df_existing is not None else pd.DataFrame()

        # 添加指数名称
        df_new["指数名称"] = name

        # 合并数据
        if df_existing is not None and len(df_existing) > 0:
            # 确保日期格式一致
            df_existing['日期'] = pd.to_datetime(df_existing['日期'])
            df_new['日期'] = pd.to_datetime(df_new['日期'])

            # 合并并去重（保留最新数据）
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['日期'], keep='last')
            df_combined = df_combined.sort_values('日期').reset_index(drop=True)
        else:
            df_combined = df_new
            df_combined['日期'] = pd.to_datetime(df_combined['日期'])
            df_combined = df_combined.sort_values('日期').reset_index(drop=True)

        # 保存数据
        df_combined.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"{name} 更新完成，新增 {len(df_new)} 行，共 {len(df_combined)} 行")
        return df_combined

    except Exception as e:
        print(f"{name} 采集失败，错误：{e}")
        return df_existing if df_existing is not None else None


def update_all_indices(force_full=False):
    """更新所有指数数据"""
    for idx_code, idx_name in index_list:
        fetch_index_data(idx_code, idx_name, force_full=force_full)
        time.sleep(1.2)  # 访问间隔，防止被限制访问
    print("\n✅全部任务结束，数据保存在 ./index_data 文件夹")


if __name__ == "__main__":
    # 正常使用：增量更新
    update_all_indices(force_full=False)

    # 如果需要强制全量更新，可以改为：
    # update_all_indices(force_full=True)