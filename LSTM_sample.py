import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#定义LSTM模型
LSTM = nn.LSTM(input_size=5, hidden_size=10, num_layers=2, batch_first=True)
x = torch.randn(3, 4, 5)
h0 = torch.randn(2, 3, 10)
c0 = torch.randn(2, 3, 10)
output, (hn, cn) = LSTM(x, (h0, c0))
# print(output)
# # print(hn.size())
# # print(cn.size())
import requests
# requests.adapters.DEFAULT_RETRIES = 5
# s = requests.session()
# s.keep_alive = False
import json
import requests

data_path = 'BTC_USD_BITFINEX.csv'
if os.path.isfile(data_path):
    data = pd.read_csv(data_path)
else:
    #爬虫需要伪装成浏览器
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1',}
    resp = requests.get('https://www.quantinfo.com/API/m/chart/history?symbol=BTC_USD_BITFINEX&resolution=60&from=1525622626&to=1562658565', headers=header, verify=False)
    data = resp.json()
    df = pd.DataFrame(data, columns=['t', 'o', 'h', 'l', 'c', 'v'])
    df.to_csv('BTC_USD_BITFINEX.csv', header=True)
print(df.head(5))

# #数据预处理
# df.index = df['t']
# df = (df-df.mean())/df.std()
# df['n'] = df['c'].shift(-1)
# df = df.dropna()
# df = df.astype(np.float32)
# # print(df[['c','n']].head(100))
#
# #准备训练数据
# seq_len = 10 #输入10个周期的数据
# train_size = 800 #训练集batch_size
# print(len(data))
# # def create_dataset(data, seq_len):
# #     dataX, dataY = [], []

