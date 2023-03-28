import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

edge_size = 256

def read_data_bananas(is_train=True):
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        #  print(img_name)
        #  print(type(img_name))
        #  print(target)
        #  print(list(target))
        #  break
        #  print(type(target))
        full_img_path = os.path.join(data_dir, 'bananas_train' if is_train else 'bananas_val', 'images', img_name)
        images.append(torchvision.io.read_image(full_img_path) / 255)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(dim=1) / edge_size

#  X, y = read_data_bananas(is_train=True)
#  print(len(X), len(y))
#  print(X.size())
#  print(y.size())
#  print(X[0])
#  print(y[0].dtype)


class BananaDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        self.features, self.labels = read_data_bananas(is_train)
        print(''.join(['read ', str(len(self.features)),' ', 'training' if is_train else 'validation', ' examples']))

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
        

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    train_set = BananaDataset(is_train=True)
    val_set = BananaDataset(is_train=False)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
    val_iter = torch.utils.data.DataLoader(val_set, batch_size, shuffle=False, num_workers=4)
    return train_iter, val_iter

batch_size = 32
train_iter, val_iter = load_data_bananas(batch_size)
X, y = next(iter(train_iter))
#  print(X.shape)
#  print(y.shape)
#  print(X)
#  print(y)

imgs = X[0:10].permute(0, 2, 3, 1) 
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, y[0:10]):
    d2l.show_bboxes(ax, [ label[0][1:5] * edge_size ], colors=['w'])

d2l.plt.show()




