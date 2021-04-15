import torch
from torch import nn
import d2lzh_pytorch as d2l


def corr2d_multi_in(X, K):
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]],
                  [[1, 2], [3, 4]]])


# print(corr2d_multi_in(X, K))


# T1 = torch.tensor([[1, 2, 3],
#                    [4, 5, 6],
#                    [7, 8, 9]])
# T2 = torch.tensor([[10, 20, 30],
#                    [40, 50, 60],
#                    [70, 80, 90]])
# print(torch.stack((T1, T2), dim=0))
# print(torch.stack((T1, T2), dim=1))
# print(torch.stack((T1, T2), dim=2))
# print(torch.cat((T1, T2), dim=0))
# print(torch.cat((T1, T2), dim=1))

def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输⼊X做互相关计算。所有结果使⽤stack函数合并在⼀起
    return torch.stack([corr2d_multi_in(X, k) for k in K])


K = torch.stack([K, K + 1, K + 2])
print(K.shape)  # torch.Size([3, 2, 2, 2])
print(corr2d_multi_in_out(X, K))
