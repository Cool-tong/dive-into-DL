import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

num_hiddens = 256
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
# lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)

num_steps = 35
# batch_size = 2
# state = None
# X = torch.rand(num_steps, batch_size, vocab_size)
# Y1, state_new1 = rnn_layer(X, state)
# print(Y1.shape, len(state_new1), state_new1[0].shape)  # torch.Size([35, 2, 256]) 1 torch.Size([2, 256])
# Y2, state_new2 = lstm_layer(X, state)
# print(Y2.shape, len(state_new2), state_new2[0].shape)  # torch.Size([35, 2, 256]) 2 torch.Size([1, 2, 256])

model = d2l.RNNModel(rnn_layer, vocab_size).to(device)
# print(d2l.predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx))

num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2  # 注意这⾥的学习率设置
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices,
                                  idx_to_char, char_to_idx, num_epochs, num_steps, lr,
                                  clipping_theta, batch_size, pred_period, pred_len, prefixes)
