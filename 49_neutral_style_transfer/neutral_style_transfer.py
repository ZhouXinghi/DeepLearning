import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
content_img = d2l.Image.open('../data/img/rainier.jpg')
#  d2l.plt.imshow(content_img)
#  d2l.plt.show()

style_img = d2l.Image.open('../data/img/autumn-oak.jpg')
#  d2l.plt.imshow(style_img)
#  d2l.plt.show()

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprecess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transfroms.Resize(image_shape), 
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)
    ])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage(img.permute(2, 0, 1))

pretrained_net = torchvision.models.vgg19(weights='DEFAULT')

style_layers, content_layers = [0, 5, 10, 19, 28], [5]
#  print(max(content_layers + style_layers) + 1)

net = nn.Sequential(*[ pretrained_net.features[i] for i in range(max(style_layers + content_layers) + 1) ])
#  print(net)

def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net(X)
        if i in content_layers:
            contents.append(X)
        if i in style_layers:
            styles.append(X)
    return contents, styles

def get_contents(content_img, image_shape, device):
    content_X = preprecess(content_img, image_shape)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(style_img, image_shape, device):
    style_X = preprecess(style_img, image_shape)
    _, styles_Y = extract_features(style_img, content_layers, style_layers)
    return style_X, styles_Y

def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y).mean()

def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y).mean()

def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

content_weight, style_weight, tv_weight = 1, 1e3, 10
def loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    contents_l = [ content_weight * content_loss(Y_hat, Y) for Y_hat, Y in zip(contents_Y_hat, contents_Y) ]
    styles_l = [ style_weight * style_loss(Y_hat, Y) for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram) ]
    tv_l = tv_weight * tv_loss(X)
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

class SynthesizedImage(nn.Module):
    def __init__(self, image_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(image_shape))

    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
