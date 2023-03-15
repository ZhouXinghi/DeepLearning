import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

# generate dataset
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

print(features.size())
print(labels.size())

#  a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
#  b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])
#  c = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])
#
#  # TensorDataset对tensor进行打包
#  train_ids = data.TensorDataset(a, b, c)
#  for x_train, y_label, z in train_ids:
#      print(x_train, y_label, z)
#
#  # dataloader进行数据封装
#  print('=' * 80)
#  train_loader = data.DataLoader(dataset=train_ids, batch_size=4, shuffle=True)
#  for i, data in enumerate(train_loader, 1):
#  # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）
#      x_data, label, z = data
#      print(' batch:{0} x_data:{1}  label: {2}  z: {3}'.format(i, x_data, label, z))

# read dataset
def load_array(data_tuple, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_tuple)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size, is_train=True)

#  print(next(iter(data_iter)))
net = nn.Sequential(nn.Linear(2, 1))

# initialize
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# define loss
loss = nn.MSELoss()

# define trainer
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#train

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

#  print(net.parameters())
for para in net.parameters():
    print(para)
