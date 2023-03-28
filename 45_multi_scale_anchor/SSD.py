import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

b1 = cls_predictor(8, 5, 10)
Y1 = b1(torch.zeros((2, 8, 20, 20)))
b2 = cls_predictor(16, 3, 10)
Y2 = b2(torch.zeros((2, 16, 10, 10)))
print(Y1.size())
print(Y2.size())

# 高宽减半块
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

print(down_sample_blk(3, 10)(torch.zeros(2, 3, 20, 20)).size())

# 基本网络块
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)

print(base_net()(torch.zeros((2, 3, 256, 256))).size())

# 5 stages
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveAvgPool2d((1, 1))
    else:
        bkl = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, sizes, ratios, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes, ratios)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

# hyperparams
sizes = [[ 0.2, 0.272 ], [ 0.37, 0.447 ], [ 0.54, 0.619 ], [ 0.71, 0.79 ], [ 0.88, 0.961 ]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
