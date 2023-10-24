import numpy as np
import pandas as pd
import torch
from sklearn import datasets
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas
from sklearn import tree
from sklearn.metrics import classification_report
path_test = 'data/test_data.csv'
path_train = 'data/train_data .csv'
test_data = pd.read_csv(path_test)
train_data = pd.read_csv(path_train)
# columns_to_fill1 = train_data.columns[:-1]
# train_data[columns_to_fill1] = train_data[columns_to_fill1].fillna(train_data[columns_to_fill1].mean())
# columns_to_fill2 = test_data.columns[:-1]
# test_data[columns_to_fill2] = test_data[columns_to_fill2].fillna(test_data[columns_to_fill2].mean())
mode_values1 = train_data.mode().iloc[0]
train_data = train_data.fillna(mode_values1)
mode_values2 = test_data.mode().iloc[0]
test_data = test_data.fillna(mode_values1)
# 20,32,54,57,60,64,65,77,78,80,88,,92,100,
# columns_to_drop = ['feature57', 'feature77', 'feature100','feature20', 'feature32', 'feature54','feature60', 'feature64', 'feature65', 'feature78', 'feature80', 'feature88', 'feature92']
columns_to_drop = ['feature57', 'feature77', 'feature100', 'feature20', 'feature60', 'sample_id']
train_data = train_data.drop(columns_to_drop, axis = 1)
test_data = test_data.drop(columns_to_drop, axis = 1)
x_train = train_data.drop('label',axis = 1)
y_train = train_data['label']
x_test = test_data.drop('label', axis = 1)
y_test = test_data['label']
x_train = torch.tensor(x_train.values).float()
x_test = torch.tensor(x_test.values).float()
y_train = torch.tensor(y_train.values).long()
y_test = torch.tensor(y_test.values).long()
class NET(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output):
        super(NET, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1).float()
        self.relu1 = nn.ReLU()

        self.hidden2 = nn.Linear(n_hidden1, n_hidden2).float()
        self.relu2 = nn.ReLU()

        self.hidden3 = nn.Linear(n_hidden2, n_hidden3).float()
        self.relu3 = nn.ReLU()

        self.hidden4 = nn.Linear(n_hidden3, n_hidden4).float()
        self.relu4 = nn.ReLU()

        self.out = nn.Linear(n_hidden4, n_output)
        self.softmax =nn.Softmax(dim = 1)
    def forward(self, x):
        hidden1 = self.hidden1(x)
        relu1 = self.relu1(hidden1)
# 完善代码:
        hidden2 = self.hidden2(relu1)
        relu2 = self.relu2(hidden2)

        hidden3 = self.hidden3(relu2)
        relu3 = self.relu3(hidden3)

        hidden4 = self.hidden4(relu3)
        relu4 = self.relu4(hidden4)

        out = self.out(relu4)
        out = self.softmax(out)

        return out
    def test(self, x):
        y_pred = self.forward(x)
        y_predict = self.softmax(y_pred)
        # _, predicted = torch.max(out, 1)
        return y_predict
net = NET(n_feature=102, n_hidden1=500, n_hidden2=500, n_hidden3=300, n_hidden4=300, n_output=6)
# optimizer = torch.optim.Adam(net.parameters(),lr = 0.1)
optimizer = torch.optim.SGD(net.parameters(),lr = 1)
loss_func = torch.nn.CrossEntropyLoss()
costs = []
epochs = 200
for epoch in range(epochs):
    cost = 0
    out = net.forward(x_train)
    loss = loss_func(out, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        out = net.forward(x_test)
        _, predicted = torch.max(out, 1)
        accuracy1 = torch.eq(predicted, y_test).sum().item() / y_test.size(0)
        print(epoch)
        print("测试集准确率为", accuracy1 * 100, "%")
y_pred = out = net.forward(x_train)
_, y_pred = torch.max(out, 1)
report = classification_report(y_test, y_pred)
print(report)


