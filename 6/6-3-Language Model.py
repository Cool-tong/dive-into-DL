import torch
import random
import zipfile
import d2lzh_pytorch as d2l

my_seq = list(range(30))
# for X, Y in d2l.data_iter_random(my_seq, batch_size=2, num_steps=6):
#     print('X: ', X, '\nY: ', Y, '\n')

for X, Y in d2l.data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')