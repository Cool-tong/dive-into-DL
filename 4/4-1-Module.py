# 可以通过继承 Module 类来构造模型。
# Sequential 、 ModuleList 、 ModuleDict 类都继承⾃ Module 类。
# 虽然 Sequential 等类可以使模型构造更加简单，但直接继承 Module 类可以极⼤地拓展模型构造的灵活性。
import torch
from torch import nn
from collections import OrderedDict


# class MLP(nn.Module):
#     def __init__(self, **kwargs):
#         super(MLP, self).__init__(**kwargs)
#         self.hidden = nn.Linear(784, 256)
#         self.act = nn.ReLU()
#         self.output = nn.Linear(256, 10)
#
#     def forward(self, x):
#         a = self.act(self.hidden(x))
#         return self.output(a)


# X = torch.rand(2, 784)
# net = MLP()
# print(net)
# print(net(X))


# class MySequential(nn.Module):
#     def __init__(self, *args):
#         super(MySequential, self).__init__()
#         if len(args) == 1 and isinstance(args[0], OrderedDict):  # 如果传入的是一个OrderedDict
#             for key, module in args[0].items():
#                 self.add_module(key, module)  # add_module⽅法会将module添加进self._modules(⼀个OrderedDict)
#         else:  # 传⼊的是⼀些Module
#             for idx, module in enumerate(args):
#                 self.add_module(str(idx), module)
#
#     def forward(self, input):
#         for module in self._modules.values():
#             input = module(input)
#         return input
#
#
# net = MySequential(
#     nn.Linear(784, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10),
# )
# print(net)
# print(net(X))

# ModuleList 接收⼀个⼦模块的列表作为输⼊，然后也可以类似List那样进⾏append和extend操作
# net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
# net.append(nn.Linear(256, 10)) # # 类似List的append操作
# print(net[-1]) # 类似List的索引访问
# print(net)

# net = nn.ModuleDict({
#     'linear': nn.Linear(784, 256),
#     'act': nn.ReLU(),
# })
# net['output'] = nn.Linear(256, 10)  # 添加
# print(net['linear'])  # 访问
# print(net.output)
# print(net)

class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = torch.rand((20, 20), requires_grad=False)  # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # nn.ReLU作为一个层结构，必须添加到nn.Module容器中才能使用
        # 而nn.functional.relu作为一个函数调用
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)
        # 复⽤全连接层，等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这⾥我们需要调⽤item函数来返回标量进⾏⽐较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


# 在这个 FancyMLP 模型中，我们使⽤了常数权 rand_weight （注意它不是可训练模型参数）、做了
# 矩阵乘法操作（ torch.mm ）并᯿复使⽤了相同的 Linear 层。下⾯我们来测试该模型的前向计算。

# X = torch.rand(2, 20)
# net = FancyMLP()
# print(net)
# print(net(X))

class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)


net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())
X = torch.rand(2, 40)
print(net)
print(net(X))
