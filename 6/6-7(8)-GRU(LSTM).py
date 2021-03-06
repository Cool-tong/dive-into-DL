import torch
from torch import nn
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

num_hiddens = 256
num_epochs, num_steps, batch_size, clipping_theta = 160, 35, 32, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
lr = 1e-2  # 注意调整学习率

# print("using GRU")
# gru_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
# model = d2l.RNNModel(gru_layer, vocab_size)

print("using LSTM")
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = d2l.RNNModel(lstm_layer, vocab_size)

d2l.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device, corpus_indices, idx_to_char,
                                  char_to_idx, num_epochs, num_steps, lr, clipping_theta, batch_size,
                                  pred_period, pred_len, prefixes)
