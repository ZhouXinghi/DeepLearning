import os
import torch
import torchvision
from d2l import torch as d2l

#  voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
#  voc_dir = '../data/VOCdevkit/VOC2012'
#  print(voc_dir)

def read_voc_images(voc_dir, is_train=True):
    """Read all VOC features and label images"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
        #  print(images)
        #  print(type(images))
        #  print(f.read())
        #  print(type(f.read()))
    features, labels = [], []
    for img_name in images:
        img_path = os.path.join(voc_dir, 'JPEGImages', f'{img_name}.jpg')
        label_path = os.path.join(voc_dir, 'SegmentationClass', f'{img_name}.png')
        feature = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        #  label = torchvision.io.read_image(label_path)
        label = torchvision.io.read_image(label_path, mode=torchvision.io.ImageReadMode.RGB)
        features.append(feature)
        labels.append(label)
    return features, labels



VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def create_rgb2index():
    rgb2index = torch.zeros(256 ** 3, dtype=torch.long)
    for i, rgb in enumerate(VOC_COLORMAP):
        r, g, b = rgb
        rgb2index[r * 256 * 256 + b * 256 + g] = i
    return rgb2index

def label2indices(label, rgb2index):
    rgbs = label.permute(1, 2, 0).numpy().astype('int32')
    idx = rgbs[:, :, 0] * 256 * 256 + rgbs[:, :, 1] * 256 + rgbs[:, :, 2]
    return rgb2index[idx]

def random_crop(feature, label, height, width):
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, voc_dir, is_train, crop_size):
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train)
        self.features = [ self.normalize_image(feature) for feature in self.filter(features) ]
        self.labels = [ label for label in self.filter(labels) ]
        self.rgb2index = create_rgb2index()
        print(''.join(['read', str(len(self.features)), 'examples']))
    
    def normalize_image(self, img):
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalize(img.float())

    def filter(self, imgs):
        return [ img for img in imgs if (img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1]) ]

    def __getitem__(self, idx):
        feature, label = random_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return feature, label2indices(label, self.rgb2index)

    def __len__(self):
        return len(self.features)

def load_data_voc(batch_size, crop_size):
    #  voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
    voc_dir = '../data/VOCdevkit/VOC2012'
    voc_train = VOCSegDataset(voc_dir, is_train=True, crop_size=crop_size)
    voc_valid = VOCSegDataset(voc_dir, is_train=False, crop_size=crop_size)
    
    train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True, drop_last=True, num_workers=4)
    valid_iter = torch.utils.data.DataLoader(voc_valid, batch_size, shuffle=False, drop_last=True, num_workers=4)
    return train_iter, valid_iter





#  train_features, train_labels = read_voc_images(voc_dir)
#  print(train_features[0].size())
#  print(train_labels[0].size())
#
#  imgs = train_features[0:5] + train_labels[0:5]
#  imgs = [ img.permute(1, 2, 0) for img in imgs ]
#  print(imgs[-1])
#  d2l.show_images(imgs, 2, 5)
#  d2l.plt.show()

#  y = label2indices(train_labels[0], create_rgb2index())
#  print(train_labels[0].size())
#  print(y.size())
#  print(y)


# randomly crop an image 5 times
#  imgs = []
#  for _ in range(5):
#      imgs += random_crop(train_features[0], train_labels[0], 200, 300)
#  imgs = [ img.permute(1, 2, 0) for img in imgs ]
#  d2l.show_images(imgs[::2] + imgs[1::2], 2, 5)
#  d2l.plt.show()

#  voc_train = VOCSegDataset(voc_dir, is_train=True, crop_size=(320, 480))
#  voc_valid = VOCSegDataset(voc_dir, is_train=False, crop_size=(320, 480))

#  batch_size = 64
#  train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True, drop_last=True, num_workers=4)
#  for X, y in train_iter:
#      print(X.shape)
#      print(y.shape)
#      break
