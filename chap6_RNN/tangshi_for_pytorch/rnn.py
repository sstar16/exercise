# rnn_improved.py
# 改进版本的 RNN 诗歌生成模型
# 原版本存在的问题：
#   1. forward() 中使用了 F.relu 激活后再接 log_softmax，会导致负值被截断，严重影响分布；
#   2. 定义了 self.softmax = nn.LogSoftmax() 但从未使用，真正用到的是 F.log_softmax；
#   3. main.py 中逐样本循环训练，输入形状 (seq_len,1) 与 batch_first=True 不匹配，
#      LSTM 实际将 seq_len 视为 batch 大小、将 1 视为序列长度，完全错误；
#   4. 梯度裁剪使用了已废弃的 clip_grad_norm（无下划线），应使用 clip_grad_norm_；
#   5. 生成时每步把历史全部重新喂入，效率低且会引入梯度问题。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# 1. 权重初始化工具函数
# ─────────────────────────────────────────────
def weights_init(m):
    """Xavier 均匀初始化线性层权重，偏置置零。"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        fan_in  = m.weight.data.size(1)
        fan_out = m.weight.data.size(0)
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


# ─────────────────────────────────────────────
# 2. 词嵌入层
# ─────────────────────────────────────────────
class word_embedding(nn.Module):
    """
    将词索引映射到稠密向量。

    参数
    ----
    vocab_length  : 词表大小
    embedding_dim : 嵌入维度
    """
    def __init__(self, vocab_length: int, embedding_dim: int):
        super(word_embedding, self).__init__()
        # 用 [-1, 1] 均匀分布初始化嵌入矩阵
        w_init = np.random.uniform(-1, 1, size=(vocab_length, embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_init))

    def forward(self, input_ids):
        """
        参数
        ----
        input_ids : LongTensor，形状任意

        返回
        ----
        嵌入向量，最后一维为 embedding_dim
        """
        return self.word_embedding(input_ids)


# ─────────────────────────────────────────────
# 3. LSTM 语言模型
# ─────────────────────────────────────────────
class RNN_model(nn.Module):
    """
    基于双层 LSTM 的字级语言模型。

    训练阶段：输入一首诗的前 T-1 个字，预测后 T-1 个字。
    生成阶段：逐步输入一个字，输出下一个字的概率分布。

    参数
    ----
    vocab_len      : 词表大小
    word_embedding : word_embedding 实例（共享参数）
    embedding_dim  : 嵌入维度
    lstm_hidden_dim: LSTM 隐藏层维度
    """
    def __init__(self,
                 vocab_len: int,
                 word_embedding: word_embedding,
                 embedding_dim: int,
                 lstm_hidden_dim: int):
        super(RNN_model, self).__init__()

        self.word_embedding_lookup = word_embedding   # 嵌入层
        self.vocab_length           = vocab_len
        self.embedding_dim          = embedding_dim
        self.lstm_dim               = lstm_hidden_dim

        # ── 【补全部分 1】定义双层 LSTM ──────────────────────────
        # input_size  : 每个时间步的特征维度 = embedding_dim
        # hidden_size : LSTM 隐藏单元数
        # num_layers  : 堆叠层数（2 层）
        # batch_first : True → 输入/输出形状为 (batch, seq, feature)
        # dropout     : 层间 dropout，缓解过拟合（最后一层不加）
        self.rnn_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3           # 原版无 dropout，加上后效果更好
        )
        # ─────────────────────────────────────────────────────────

        # 全连接层：将 LSTM 输出映射到词表大小
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)

        # 对线性层做 Xavier 初始化
        self.apply(weights_init)

    # ── 训练前向传播 ────────────────────────────────────────────
    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        批量前向传播（训练时使用）。

        参数
        ----
        sentence : LongTensor，形状 (batch_size, seq_len)
                   每行是一首诗的词索引序列（已 pad 到相同长度）

        返回
        ----
        log_probs : FloatTensor，形状 (batch_size * seq_len, vocab_len)
                    每个时间步的对数概率，供 NLLLoss 使用
        """
        # 1. 词嵌入：(batch, seq_len) → (batch, seq_len, embedding_dim)
        embed = self.word_embedding_lookup(sentence)

        # ── 【补全部分 2】LSTM 前向传播 ─────────────────────────
        # 隐藏状态初始化为 0（在当前设备上）
        device    = next(self.parameters()).device
        batch_sz  = embed.size(0)
        h0 = torch.zeros(2, batch_sz, self.lstm_dim, device=device)
        c0 = torch.zeros(2, batch_sz, self.lstm_dim, device=device)

        # output : (batch, seq_len, lstm_hidden_dim)
        # hn, cn : (num_layers, batch, lstm_hidden_dim)
        output, (hn, cn) = self.rnn_lstm(embed, (h0, c0))
        # ─────────────────────────────────────────────────────────

        # 2. 展平为 (batch * seq_len, lstm_hidden_dim) 后接全连接
        out = output.contiguous().view(-1, self.lstm_dim)   # (N, hidden)
        out = self.fc(out)                                   # (N, vocab_len)

        # 3. 对数 Softmax（注意：原版错误地在此之前加了 F.relu，会截断负数）
        log_probs = F.log_softmax(out, dim=1)               # (N, vocab_len)
        return log_probs

    # ── 逐步生成（生成时使用）──────────────────────────────────
    def generate_step(self,
                      token: torch.Tensor,
                      hidden: tuple) -> tuple:
        """
        单步生成：输入一个词，输出下一个词及更新后的隐藏状态。

        参数
        ----
        token  : LongTensor，形状 (1, 1)，当前输入词的索引
        hidden : (h, c) 元组，LSTM 隐藏状态

        返回
        ----
        next_token : LongTensor，形状 ()，预测的下一个词索引
        hidden     : 更新后的 (h, c)
        """
        # (1, 1) → (1, 1, embedding_dim)
        embed = self.word_embedding_lookup(token)

        # output : (1, 1, lstm_hidden_dim)
        output, hidden = self.rnn_lstm(embed, hidden)

        # 取最后（也是唯一的）时间步输出
        out       = self.fc(output.squeeze(1))      # (1, vocab_len)
        log_probs = F.log_softmax(out, dim=1)
        next_token = torch.argmax(log_probs, dim=1) # (1,)
        return next_token, hidden

    def init_hidden(self, batch_size: int = 1) -> tuple:
        """初始化 LSTM 隐藏状态为零张量。"""
        device = next(self.parameters()).device
        h = torch.zeros(2, batch_size, self.lstm_dim, device=device)
        c = torch.zeros(2, batch_size, self.lstm_dim, device=device)
        return (h, c)