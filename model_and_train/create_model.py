import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler

# 当前脚本所在文件夹 model_and_train
base_dir = os.path.dirname(__file__)
# 往上跳一级，定位项目根目录，再拼接 quotes
csv_path = os.path.join(base_dir, "..", "quotes", "510330.SH.csv")
# 标准化路径，消除..符号
csv_path = os.path.abspath(csv_path)
# data = pd.read_csv("C:/Users/Windows/PycharmProjects/StockRNN/quotes/510330.SH.csv")
data = pd.read_csv(csv_path)
data= data["close"].values.reshape(-1,1) # 变成多行一列的数组
# 2.归一化
scalar = MinMaxScaler()
data = scalar.fit_transform(data)

# 3.创建滑动窗口，用过去60天的数据预测后一天
def create_sequences(data,lookback=60):
    x,y = [],[]
    for i in range(lookback,len(data)):
        x.append(data[i-lookback:i,0])
        y.append(data[i,0])
    return np.array(x),np.array(y)
x,y = create_sequences(data)

# 4.划分训练集和测试集
train_size = int(len(x)*0.8)
x_train,y_train = x[:train_size],y[:train_size]
x_test,y_test = x[train_size:],y[train_size:]

# 5.将数据转化为pytorch张量，并调整为（样本数、时间步、特征数）
x_train = torch.tensor(x_train,dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train,dtype=torch.float32).unsqueeze(-1)
x_test = torch.tensor(x_test,dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test,dtype=torch.float32).unsqueeze(-1)

# 创建模型
class StockRnn(nn.Module):
    pass
