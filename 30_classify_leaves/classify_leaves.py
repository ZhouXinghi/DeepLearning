import torch
from torch import nn
from torch.utils import data
from torchvision.io import read_image
from torchvision import transforms
from torchvision import models
import pandas as pd
import numpy as np
import os
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns


csv_data = pd.read_csv('data/classify-leaves/train.csv')

n_train = int(len(csv_data) * 0.8)
n_valid = len(csv_data) - n_train
batch_size, lr, weight_decay, num_epochs = 8, 3e-4, 1e-3, 50
model_path = './pre_res_model.ckpt'

print(n_train)
#  print(csv_data.head(5))
#  print(csv_data.describe())
#  print(len(csv_data))
#  print(csv_data.iloc[6, 0], csv_data.iloc[6, 1])

#  def barw(ax):
#      for p in ax.patches:
#          val = p.get_width()
#          x = p.get_x() + p.get_width()
#          y = p.get_y() + p.get_height() / 2
#          ax.annotate(round(val, 2), (x, y))
#
#  plt.figure(figsize=(15, 30))
#  ax0 = sns.countplot(y=labels_data_frame['label'], order=labels_data_frame['label'].value_counts().index)
#  barw(ax0)
#  plt.show()

leaves_labels = sorted(list(set(csv_data['label'])))
n_classes = len(leaves_labels)
#  print(n_classes)
#  print(leaves_labels[:10])

#  class_to_num = { c: l for (c, l) in zip(leaves_labels, range(n_classes)) }
class_to_num = { label: i for (i, label) in enumerate(leaves_labels) }
#  print(len(class_to_num))
#  for i, item in enumerate(class_to_num.items()):
#      print(item)
#      if i == 10:
#          break

num_to_class = { i: label for (i, label) in enumerate(leaves_labels) }
#  print(len(num_to_class))
#  for i, item in enumerate(class_to_num.items()):
#      print(item)
#      if i == 10:
#          break

class LeavesData(data.Dataset):
    def __init__(self, csv_path, img_dir):
        self.img_dir = img_dir
        self.csv_data = pd.read_csv(csv_path)
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.csv_data.iloc[index, 0])
        
        image = transforms.ToTensor()(Image.open(img_path))
        #  image = read_image(img_path)
        #  image.dtype = torch.float32
        label = class_to_num[self.csv_data.iloc[index, 1]]
        return image, label
    def __len__(self):
        return len(self.csv_data)

all_dataset = LeavesData('data/classify-leaves/train.csv', 'data/classify-leaves')
train_dataset, valid_dataset = data.random_split(all_dataset, [n_train, n_valid])
#  print(len(train_dataset), len(valid_dataset), len(all_dataset))

#  img, label = train_dataset[0]
#  print(img.size())
#  transforms.ToPILImage()(img).show()
#  print(label)

train_loader = data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
valid_loader = data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=2)
#  for i, (X, y) in enumerate(train_loader):
#      for j in range(X.shape[0]):
#          transforms.ToPILImage()(X[j]).show()
#          print(y[j])
#      if i == 0:
#          break

net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
net.fc.out_features = n_classes

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

net.fc.apply(init_weights)
#  print(net.fc)
#  print(net)
#  for i, (X, y) in enumerate(train_loader):
    #  print(X)
    #  print(net(X).size())
    #  print(y)
    #  break
#  X = next(iter(train_loader))
#  print(X.size())
#  X = net(X)
#  print(X.size())
#  print(net)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

def evaluate(net, valid_loader, loss, device):
    net.to(device)
    net.eval()
    valid_loss_list = []
    valid_acc_list = []
    for i_batch, (X, y) in enumerate(valid_loader):
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y_hat = net(X)
        l = loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        valid_loss_list.append(l.item())
        valid_acc_list.append(acc)
    valid_loss = sum(valid_loss_list) / len(valid_loss_list)
    valid_acc = sum(valid_acc_list) / len(valid_acc_list)
    return valid_loss, valid_acc


def train_model(net, train_loader, valid_loader, loss, num_epochs, optimizer, device):
    print('training on', device)
    net.to(device)

    best_acc = 0.0
    for epoch in range(num_epochs):
        # train
        net.train()
        train_loss_list = []
        train_acc_list = []
        for i_batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            acc = (y_hat.argmax(dim=1) == y).float().mean()
            train_loss_list.append(l.item())
            train_acc_list.append(acc)
            print(f'batch {i_batch}: loss {l.item():.5f}, acc {acc:.5f}')
        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_acc = sum(train_acc_list) / len(train_acc_list)
        print(f'[ Train | epoch{epoch + 1}/{num_epochs} ]\ttrain_loss:{train_loss:.5f}, train_acc:{train_acc:.5f}')

        # validation
        valid_loss, valid_acc = evaluate(net, valid_loader, loss, device)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(net.state_dict(), model_path)
            print(f'saving model with acc {best_acc:.5f}')
            
#  train_model(net, train_loader, valid_loader, loss, num_epochs, optimizer, device=torch.device('cuda:0'))
net.load_state_dict(torch.load('30_classify_leaves/models/pre_res_model_0.85866.ckpt'))
loss, acc = evaluate(net, valid_loader, loss, device=torch.device('cuda:0'))
print(loss, acc)

