import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(in_channels, out_channels, num_residuals, is_first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not is_first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

if __name__ == "__main__":
    #  blk = Residual(3, 6, use_1x1conv=True, stride=2)
    #  X = torch.rand((4, 3, 6, 6))
    #  Y = blk(X)
    #  print(Y.shape)
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b2 = resnet_block(64, 64, 2, is_first_block=True)
    b3 = resnet_block(64, 128, 2)
    b4 = resnet_block(128, 256, 2)
    b5 = resnet_block(256, 512, 2)
    net = nn.Sequential(
        b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10)
    )
    #  print(net)
    
    X = torch.rand(size=(128, 1, 96, 96))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'outputshape:\t', X.shape)

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
