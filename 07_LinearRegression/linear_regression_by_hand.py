import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    """ create y = Xw + b + eps """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

#  show data

print('features:', features[0], '\nlabel:', labels[0])
d2l.set_figsize()

d2l.plt.scatter(
    features[:, 0].detach().numpy(),
    labels.detach().numpy(),
    1
)
d2l.plt.scatter(
    features[:, 1].detach().numpy(),
    labels.detach().numpy(),
    1
)
d2l.plt.show()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    #  print(indices)
    random.shuffle(indices)
    #  print(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


# initialize model parameters
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# define model
def linreg(X, w, b):
    return X @ w + b

# define loss func
def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape)) ** 2).mean()

# define optimization method
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

# Train
lr = 0.01
num_epochs = 50
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l)}')



print(w)
print(b)
