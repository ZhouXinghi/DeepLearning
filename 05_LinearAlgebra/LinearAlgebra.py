import torch

x = torch.arange(4)
print(len(x))
print(x.shape)

A = torch.arange(20, dtype=torch.float32).reshape(5,4)
B = A
#  B = A.clone()
print(id(A))
print(id(B))
print(id(A + B))

