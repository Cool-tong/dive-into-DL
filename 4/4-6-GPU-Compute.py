import torch
from torch import nn

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # 返回gpu名字，设备索引默认从0开始
print(torch.cuda.current_device())  # 返回当前设备索引
