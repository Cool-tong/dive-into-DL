import torch
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


# layer = CenteredLayer()
# print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))

# net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# y = net(torch.rand(4, 8))
# print(y.mean().item())


class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x


net = MyDense()
print(net)
