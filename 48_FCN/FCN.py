import torch
import torchvision 
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import semantic_segmentation as S

pretrained_net = torchvision.models.resnet18(weights='DEFAULT')
#  print(list(pretrained_net.children()))
#  print(list(pretrained_net.children())[-3:])
#  print(list(pretrained_net.children())[:-1])

# channel: 3 -> 512
# mapsize /= 32
net = nn.Sequential(*list(pretrained_net.children())[:-2])

#  X = torch.rand(size=(1, 3, 320, 480))
#  print(net(X).shape)

num_classes = 21
net.add_module('1x1_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transposed_conv', nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16))

#  print(net(X).shape)

############################################################### initialize transposed_conv
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

W = bilinear_kernel(num_classes, num_classes, 64)
net.transposed_conv.weight.data.copy_(W)

######################################### bilinear upsampling experiment
#  up_conv = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
#  up_conv.weight.data.copy_(bilinear_kernel(3, 3, 4))     # 直接赋值不行吗
#  #  up_conv.weight.data = bilinear_kernel(3, 3, 4)
#
#  img = torchvision.io.read_image('../data/img/catdog.jpg')
#  X = img.unsqueeze(0) / 255
#  Y = up_conv(X)
#
#  d2l.set_figsize()
#  #  d2l.plt.imshow(img.permute(1, 2, 0))
#  #  d2l.plt.imshow(X[0].permute(1, 2, 0))
#  #  d2l.plt.show()
#  #  d2l.plt.imshow(Y[0].detach().permute(1, 2, 0))
#  imgs = [ img.permute(1, 2, 0), Y[0].detach().permute(1, 2, 0) ]
#  print(imgs[0].shape, imgs[1].shape)
#  d2l.show_images(imgs, 1, 2)
#  d2l.plt.show()

####################################### read data
batch_size=32
crop_size=(320, 480)
train_iter, valid_iter = S.load_data_voc(batch_size, crop_size)


####################################### train
def loss(input, target):
    return F.cross_entropy(input, target, reduction='none').mean(dim=1).mean(dim=1)
#  loss = nn.CrossEntropyLoss(reduction='none')
X, label = next(iter(train_iter))

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, valid_iter, loss, trainer, num_epochs, devices)

########################################  predict
def predict(img):
    X = valid_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
