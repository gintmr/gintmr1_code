import os
import numpy as np
from PIL import Image
import numpy as np
import torch
from sklearn import datasets
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def load_images(folder_path):
    images = []
    labels = []
    for folder_name in os.listdir(folder_path):
        folder_dir = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_dir):
            for file_name in os.listdir(folder_dir):
                file_path = os.path.join(folder_dir, file_name)
                if file_path.endswith(".png") or file_path.endswith(".jpg"):
                    try:
                        # image = Image.open(file_path).convert('L')
                        # image = image.resize((1, 28*28))
                        # image_array = np.array(image)
                        # images.append(image_array)
                        # labels.append(folder_name)
                        image = Image.open(file_path).convert('L')
                        image = image.resize((28, 28))
                        image_array = np.array(image).astype(float)
                        image_array /= 255.0
                        images.append(image_array.reshape(1, -1))
                        labels.append(folder_name)
                    except Exception as e:
                        print(f"Error processing image: {file_path}")
                        print(e)
    # return np.array(images), np.array(labels)
    return np.concatenate(images, axis = 0), np.array(labels)


folder_path1 = './data'

# 加载图片并进行分类
images, labels = load_images(folder_path1)


def load_images(folder_path):
    images = []
    labels = []
    for folder_name in os.listdir(folder_path):
        folder_dir = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_dir):
            for file_name in os.listdir(folder_dir):
                file_path = os.path.join(folder_dir, file_name)
                if file_path.endswith(".png") or file_path.endswith(".jpg"):
                    try:
                        # image = Image.open(file_path).convert('L')
                        # image = image.resize((1, 28*28))
                        # image_array = np.array(image)
                        # images.append(image_array)
                        # labels.append(folder_name)
                        image = Image.open(file_path).convert('L')
                        image = image.resize((28, 28))
                        image_array = np.array(image).astype(float)
                        image_array /= 255.0
                        images.append(image_array.reshape(1, -1))
                        labels.append(folder_name)
                    except Exception as e:
                        print(f"Error processing image: {file_path}")
                        print(e)
    # return np.array(images), np.array(labels)
    return np.concatenate(images, axis = 0), np.array(labels)
folder_path2 = './data_pred'

# 加载图片并进行分类
images_test, labels_test = load_images(folder_path2)

class NET(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_hidden6, n_output):
        super(NET, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden1)
        self.relu1 = nn.ReLU()

        self.hidden2 =nn.Linear(n_hidden1, n_hidden2)
        self.relu2 = nn.ReLU()

        self.hidden3 =nn.Linear(n_hidden2, n_hidden3)
        self.relu2 = nn.ReLU()

        self.hidden4 =nn.Linear(n_hidden3, n_hidden4)
        self.relu2 = nn.ReLU()

        self.hidden5 = nn.Linear(n_hidden4, n_hidden5)
        self.hidden6 = nn.Linear(n_hidden5, n_hidden6)

        self.out = nn.Linear(n_hidden6, n_output)
        self.softmax =nn.Softmax(dim = 1)

    def forward(self, x):
        hidden1 = self.hidden1(x)
        relu1 = self.relu1(hidden1)

        hidden2 = self.hidden2(relu1)
        relu2 = self.relu2(hidden2)

        hidden3 = self.hidden3(relu2)
        relu3 = self.relu2(hidden3)

        hidden4 = self.hidden4(relu3)
        relu4 = self.relu2(hidden4)

        hidden5 = self.hidden5(relu4)
        relu5 = self.relu2(hidden5)

        hidden6 = self.hidden6(relu5)
        relu6 = self.relu2(hidden6)

        out = self.out(relu6)
        # out = self.softmax(out)

        return out
#测试函数
    def test(self, x):
        y_pred = self.forward(x)
        y_predict = self.softmax(y_pred)

        return y_predict
labels = labels.astype(np.int64)
labels_test = labels_test.astype(np.int64)
images = images.astype(float)
images_test = images_test.astype(float)
labels = torch.from_numpy(labels)
labels_test = torch.from_numpy(labels_test)
images = torch.from_numpy(images).float()
images_test = torch.from_numpy(images_test).float()
# images = torch.LongTensor(images)
# images_test = torch.LongTensor(images_test)

net = NET(n_feature=28*28, n_hidden1=900, n_hidden2=600, n_hidden3=360, n_hidden4=200, n_hidden5=200, n_hidden6=64, n_output=10)
# optimizer = torch.optim.SGD(net.parameters(),lr = 0.05)
optimizer = torch.optim.Adam(net.parameters(),lr = 0.05)
loss_func = torch.nn.CrossEntropyLoss()
costs = []
epochs = 2000
for epoch in range(epochs):
    cost = 0
    out = net.forward(images)
    loss = loss_func(out, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        out = net.forward(images_test)
        _, predicted = torch.max(out, 1)
        accuracy1 = torch.eq(predicted, labels_test).sum().item() / labels_test.size(0)
        print(epoch)
        print("测试集准确率为", accuracy1 * 100, "%")
y_pred = out = net.forward(images)
_, y_pred = torch.max(out, 1)
report = classification_report(labels_test, y_pred)
print(report)


#
# # 打印分类结果
# for i in range(len(images)):
#     print(f"Image: {i+1}, Label: {labels[i]}")
#     print(images[i])  # 打印图片数组
#     print()
