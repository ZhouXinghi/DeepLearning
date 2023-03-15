import torch

x = torch.arange(4.0)
print(x)
x.requires_grad_(True)
print(x.grad)

y = 2 * x @ x
#  y = 2 * torch.dot(x, x)
print(y)
y.backward()
print(x.grad)

x.grad.zero_()
print(x.grad)

y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = u @ x
z.backward()
print(x.grad == u)



