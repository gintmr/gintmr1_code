import os
import torch.nn as nn
import torch.utils.data
from torch.utils.data import random_split
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision import transforms as transforms
root_dir = "./data"
# rotation = transforms.RandomApply([transforms.RandomRotation(30)], p=0.05)
class Mnist_data(nn.Module):
    def __init__(self,data_root,batch_size=32):
        super(Mnist_data, self).__init__()
        self.batch_size = batch_size
        self.data_root =data_root
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor()
        # ])#change the image format
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.05),
            #  transforms.RandomVerticalFlip(p=0.05),
            # transforms.RandomRotation(30, p=0.1),
            # rotation,
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), 
            transforms.Resize((84, 84)),
            transforms.Grayscale(),
            transforms.ToTensor()

])

        Mnistdataset= datasets.ImageFolder(root=root_dir,transform=transform)
        # Mnistdataset = Mnistdataset.permute(1, 0, 2, 3)
        class_names = Mnistdataset.classes
        self.num = len(Mnistdataset)
        self.divide = {'train':0.85,
                       'test':0.15,
                       'val':0.0}
        self.num_samples = len(Mnistdataset)

        self.num_train = int(self.num * self.divide['train'])
        self.num_test = self.num_samples - self.num_train
        self.num_val = self.num_samples - self.num_train - self.num_test

        print(class_names)
        print('\n')
        print(len(class_names))
        print(Mnistdataset.class_to_idx)

        train_dataset, test_dataset = torch.utils.data.random_split(Mnistdataset, [self.num_train, self.num_test])

        self.train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = DataLoader(test_dataset, batch_size=self.batch_size,shuffle=False)
  #      self.val_dataset = DataLoader(val_dataset, batch_size=self.batch_size,shuffle=False)

    def train_dataloader(self):
        return self.train_dataset

    def test_dataloader(self):
        return self.test_dataset

  #  def val_dataloader(self):
  #      return self.val_dataset

    def forward(self):
        return self.train_dataset,self.test_dataset#,self.val_dataset

class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, padding=0),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 6,  4, 1, padding=0),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(6, 6,  4, 1, padding=0),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.ReLU()
        )
        self.ffn1 = nn.Sequential(
            nn.Linear( 2166, 240),
            nn.ReLU()
            # the following code should be tried
            # nn.Linear(120, 25),
            # nn.ReLu(),
            # nn.Linear(25, 5)
        )
        self.activate1 = nn.ReLU()
        self.ffn2 = nn.Sequential(
            nn.Linear(240, 64),
            nn.ReLU()
        )
        self.activate2 = nn.ReLU()
        self.ffn3 = nn.Sequential(
            nn.Linear(64, 10),
            nn.ReLU()
        )

        # self.activate3 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.flatten(start_dim=1)
        out = self.ffn1(out)
        out = self.activate1(out)
        out = self.ffn2(out)
        out = self.activate2(out)
        out = self.ffn3(out)

        return out

learning_rate = 0.06
dataset = Mnist_data(root_dir)
# dataset = dataset.permute(1,0,2,3)
train_dataloader = dataset.train_dataloader()
test_dataloader = dataset.test_dataloader()
criterion = nn.CrossEntropyLoss()

model=Mymodel()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
parameters = model.parameters()
# for parm in parameters:
#   print(parm)
epoches_num = 25
for epoch in range(epoches_num):
 model.train()
 for images, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

 model.eval()
 correct = 0
 total = 0
 with torch.no_grad():
       for images, labels in test_dataloader:
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
 accuracy = 100 * correct / total
 print(f"Test Accuracy: {accuracy}%")
 print(f"Epoch [{epoch+1}/{epoches_num}], Loss: {loss.item()}")
 loss_history = []
 accuracy_history = []
 loss_history.append(loss)
 accuracy_history.append(accuracy)
