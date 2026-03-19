import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn as rnn_lstm
start_token = 'G'
end_token = 'E'
batch_size = 64
poems = []
with open('./poems.txt', "r", encoding='utf-8', ) as f:
    print("正在读取诗歌数据...")
    for line in f.readlines():
        try:
            #print('1')
            parts = line.strip().split(':', 1)
            if len(parts) != 2:
                continue
            title, content = parts
            #print(title)
            content = content.replace(' ', '').replace('，','').replace('。','')
            content = content.replace(' ', '')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                            start_token in content or end_token in content:
                continue
            if len(content) < 5 or len(content) > 80:
                continue
            content = start_token + content + end_token
            poems.append(content)
        except ValueError as e:
            print("error1")
            pass