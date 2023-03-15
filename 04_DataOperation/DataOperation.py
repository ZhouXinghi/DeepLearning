import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())  # number of elements

X = x.reshape(3, 4)
print(X)

print(torch.zeros((2, 3, 4)))

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(torch.cat((X, X), dim=0))
print(torch.cat((X, X), dim=1))
