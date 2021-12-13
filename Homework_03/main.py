import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn as nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from data import MyDataset

import dataprocess
import dataloader
from model import LSTM_Net
from train import training
from test import testing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sen_len = 30 #定义句子长度为30
fix_embedding = True # 训练时固定embedding
batch_size = 128
epoch = 5
lr = 0.001
model_dir = "./model/"
# 划分数据
data_process = dataloader.DataLoader(r"H:\个人资料\研究生\研究生课件\深度学习\作业\第三次\weibo_senti_100k\\", sen_len)
contents = data_process.loadData()
train_val_data, train_val_label, test_data, test_label = data_process.divideChineseData(contents)
# print(len(train_val_data))
# print(len(test_data))
# data_process.train_word2vec(train_data+val_data+test_data)
# 生成词向量
data_process.loadSentences(train_val_data)
train_val_embedding = data_process.make_embedding(load=True)
# print(train_val_embedding)
train_val_data = data_process.sentence_word2idx()
train_val_label = data_process.labels_to_tensor(train_val_label)

model = LSTM_Net(train_val_embedding, embedding_dim=250, hidden_dim=300, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device)
num = len(train_val_data)
train_data, val_data = train_val_data[:int(0.8*num)], train_val_data[int(0.8*num):]
train_label, val_label = train_val_label[:int(0.8*num)], train_val_label[int(0.8*num):]
train_dataset = MyDataset(X=train_data, y=train_label)
val_dataset = MyDataset(X=val_data, y = val_label)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0)   # windows系统下不要用多线程，linux系统随意！

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0)

#开始训练
print("Begin training ......................")
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)
# 开始测试

# 测试数据
test_str = []
for i in range(10):
    cut_list = test_data[i]
    cut_str = ""
    for word in cut_list:
        cut_str = cut_str + word
    test_str.append(cut_str)
for i in range(10):
    print(test_str[i])
    print("\n")
test_process = dataprocess.Preprocess(test_data, sen_len)
test_embedding = test_process.make_embedding(load=True)
test_data = test_process.sentence_word2idx()
test_dataset = MyDataset(X=test_data, y=None)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0)
print("Begin testing..................")
test_model = torch.load("./model/ckpt.model")
outputs = testing(batch_size, test_loader, test_model, device)
print(outputs[:10])





