import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn import datasets
import torch.nn.functional as Fun
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch

# 数据准备
# dataset = datasets.load_iris()
# dataut = dataset['data']
# priciple = dataset['target']
# input = torch.FloatTensor(dataset['data'])
# label = torch.LongTensor(dataset['target'])
# print(input)
# print(label)
data = pd.read_csv("E:\\data\\result\\dataset_random_split1_modified.csv")
n_feature = data.shape[1] - 1
n_output = data['Label'].value_counts().shape[0]
n_hidden = 2 * n_feature
print(n_feature, n_hidden, n_output)
data_x = data[data.columns[:-1]].to_numpy()
data_y = data['Label']
encoder = LabelEncoder()
data_y = encoder.fit_transform(data_y)
input = torch.FloatTensor(data_x)
label = torch.LongTensor(data_y)


# 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # activation function for hidden layer we choose sigmoid
        x = self.out(x)
        return x


net = Net(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # SGD: 随机梯度下降
loss_func = torch.nn.CrossEntropyLoss()  # 针对分类问题的损失函数

# 训练数据
for t in range(500):
    out = net(input)  # input x and predict based on x
    loss = loss_func(out, label)  # 输出与label对比
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # back propagation, compute gradients
    optimizer.step()  # apply gradients

out = net(input)  # out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
prediction = torch.max(out, 1)[1]  # 1返回index  0返回原值
pred_y = prediction.data.numpy()
target_y = label.data.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print("accuracy", accuracy)
