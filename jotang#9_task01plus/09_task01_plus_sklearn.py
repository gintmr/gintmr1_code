import numpy as np
import pandas as pd
import torch
from sklearn import datasets
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
# import graphviz
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
path_test = 'data/test_data.csv'
path_train = 'data/train_data .csv'
test_data = pd.read_csv(path_test)
train_data = pd.read_csv(path_train)
# # 获取除了最后一列的列名
# columns_to_fill1 = train_data.columns[:-1]
# # 使用每一列（除了最后一列）的平均值填充空缺值
# train_data[columns_to_fill1] = train_data[columns_to_fill1].fillna(train_data[columns_to_fill1].mean())
# columns_to_fill2 = test_data.columns[:-1]
# # 使用每一列（除了最后一列）的平均值填充空缺值
# test_data[columns_to_fill2] = test_data[columns_to_fill2].fillna(test_data[columns_to_fill2].mean())
#寻找众数
mode_values1 = train_data.mode().iloc[0]
# 填充缺失值
train_data = train_data.fillna(mode_values1)
#寻找众数
mode_values2 = test_data.mode().iloc[0]
# 填充缺失值
test_data = test_data.fillna(mode_values1)
# 20,32,54,57,60,64,65,77,78,80,88,,92,100,
# columns_to_drop = ['feature57', 'feature77', 'feature100','feature20', 'feature32', 'feature54','feature60', 'feature64', 'feature65', 'feature78', 'feature80', 'feature88', 'feature92']
columns_to_drop = ['feature57', 'feature77', 'feature100']
train_data = train_data.drop(columns_to_drop, axis = 1)
test_data = test_data.drop(columns_to_drop, axis = 1)
x_train = train_data.drop('label',axis = 1)
y_train = train_data['label']
x_test = test_data.drop('label', axis = 1)
y_test = test_data['label']
# clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = RandomForestClassifier()
clf = clf.fit(x_train, y_train)# 使用实例化好的模型进行拟合操作
score = clf.score(x_test, y_test) #返回预测的准确度
print(score)

y_pred = clf.predict(x_test)
report = classification_report(y_test, y_pred)
# 输出分类报告
print(report)

# param_grid = {
#     'max_depth': [16, 8, 4, 6, 7, 10, 20, 3, 5],
#     'min_samples_split': [3, 5, 1, 2, 4, 6, 8, 9],
#     'min_samples_leaf': [8, 2, 4, 1, 5, 6, 10 ],
#     'max_features': [None, 'sqrt', 'log2']
# }
# tree_classifier = DecisionTreeClassifier()
# tree_classifier
# grid_search = GridSearchCV(estimator=tree_classifier, param_grid=param_grid, cv=5)
# grid_search.fit(x_train, y_train)
# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)
# clf_best = DecisionTreeClassifier(max_depth=10, max_features=None, min_samples_leaf=10, min_samples_split=3)
# clf_best = clf_best.fit(x_train, y_train)
# score1 = clf_best.score(x_test, y_test)
# print(score1)
