import torch

def func(x):
    x -= x.mean()
    return x

def func2(x):
    x += torch.tensor([1, 2, 3, 4])

if __name__ == "__main__":
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    func(x)
    print(x)

    #  y = x
    #  func2(y)
    #  print(x)
    #  print(y)
