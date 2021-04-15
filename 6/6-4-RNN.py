import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# 读取周杰伦专辑歌词数据集
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

# X = torch.arange(10).view(2, 5)
# inputs = d2l.to_onehot(X, vocab_size)
# print(len(inputs))  # 5
# print(inputs[0].shape)  # torch.Size([2, 1027])

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))
    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


# state = init_rnn_state(X.shape[0], num_hiddens, device)
# inputs = d2l.to_onehot(X.to(device), vocab_size)
# params = get_params()
# outputs, state_new = rnn(inputs, state, params)
# print(len(outputs))
# print(outputs[0].shape)
# print(state_new[0].shape)

# print(d2l.predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens,
#                       vocab_size, device, idx_to_char, char_to_idx))

num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

# 采用随机采样训练模型并创作歌词
# d2l.train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, device,
#                       corpus_indices, idx_to_char, char_to_idx, True, num_epochs, num_steps,
#                       lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
# 采⽤相邻采样训练模型并创作歌词
d2l.train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, device,
                      corpus_indices, idx_to_char, char_to_idx, False, num_epochs, num_steps,
                      lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
