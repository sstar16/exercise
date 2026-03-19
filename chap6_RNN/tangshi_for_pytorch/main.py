# main_improved.py
# 改进版主训练与生成脚本
#
# 原版关键缺陷（已全部修复）：
#   ① 输入形状错误：原版 np.expand_dims(x, axis=1) → (seq_len, 1)，
#      LSTM(batch_first=True) 会把 seq_len 当 batch、1 当 seq_len，完全混乱。
#      修复：改为真正的批量 padding，输入形状 (batch, seq_len)。
#   ② 逐样本循环：原版 for index in range(BATCH_SIZE) 逐条训练，极慢且无法利用并行。
#      修复：整批数据打包成 tensor 一起送入模型。
#   ③ clip_grad_norm 废弃：改为 clip_grad_norm_（带下划线的 in-place 版本）。
#   ④ 生成时每步喂入全部历史：效率低，改为维护隐藏状态逐步生成。

import numpy as np
import collections
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import rnn as rnn_module

# ─── 全局常量 ────────────────────────────────────────────────
START_TOKEN = 'G'   # 序列起始标记
END_TOKEN   = 'E'   # 序列结束标记
BATCH_SIZE  = 64    # 训练批大小
EPOCHS      = 100    # 训练轮数
LR          = 0.001 # 学习率
MAX_LEN     = 80    # 诗的最大字数（过长则跳过）
EMBED_DIM   = 128   # 词嵌入维度（原版 100，适当加大）
HIDDEN_DIM  = 256   # LSTM 隐藏维度（原版 128，适当加大）
GRAD_CLIP   = 1.0   # 梯度裁剪阈值


# ─────────────────────────────────────────────
# 数据预处理
# ─────────────────────────────────────────────
def process_poems(file_name: str):
    """
    读取诗歌文件，构建词表并将诗歌转换为索引序列。

    格式：每行 "标题:内容"（poems.txt）

    返回
    ----
    poems_vector : List[List[int]]，每首诗的词索引列表
    word_int_map : Dict[str, int]，词到索引的映射
    words        : Tuple[str]，索引到词的映射（按索引排列）
    """
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f:
            try:
                parts = line.strip().split(':', 1)
                if len(parts) != 2:
                    continue
                _, content = parts
                # 去除空格和标点，保留汉字
                #content = content.replace(' ', '').replace('，', '').replace('。', '')
                # 过滤含有特殊字符的诗
                if any(c in content for c in ['_', '(', '（', '《', '[', START_TOKEN, END_TOKEN]):
                    continue
                if len(content) < 5 or len(content) > MAX_LEN:
                    continue
                # 添加起止标记
                poems.append(START_TOKEN + content + END_TOKEN)
            except Exception:
                pass

    # 按长度排序（有利于 batch 内长度均匀，减少 padding）
    poems.sort(key=len)

    # 统计词频并构建词表
    counter = collections.Counter(c for poem in poems for c in poem)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = words + (' ',)                              # 末尾加 padding 符
    word_int_map = {w: i for i, w in enumerate(words)}

    # 将每首诗转为索引序列
    poems_vector = [[word_int_map[c] for c in poem] for poem in poems]
    return poems_vector, word_int_map, words


# ─────────────────────────────────────────────
# Batch 生成（支持变长序列 padding）
# ─────────────────────────────────────────────
def generate_batch(poems_vec, word_to_int, batch_size: int):
    """
    将诗歌序列列表分成若干批，并做 padding。

    修复点：原版用 pad=space_index 手动补齐，这里用 PyTorch 内置 pad_sequence。

    返回
    ----
    每批 (x_batch, y_batch)：
        x_batch : LongTensor (batch, max_seq_len)，输入序列（去掉最后一个字）
        y_batch : LongTensor (batch, max_seq_len)，目标序列（去掉第一个字）
    """
    pad_idx = word_to_int.get(' ', 0)   # padding 使用空格的索引
    n_chunk = len(poems_vec) // batch_size

    for i in range(n_chunk):
        batch = poems_vec[i * batch_size: (i + 1) * batch_size]

        # 构建输入 x（去掉末尾）和目标 y（去掉开头）
        x_seqs = [torch.tensor(poem[:-1], dtype=torch.long) for poem in batch]
        y_seqs = [torch.tensor(poem[1:],  dtype=torch.long) for poem in batch]

        # pad_sequence 默认 batch_first=False，设为 True 得到 (batch, max_len)
        x_pad = torch.nn.utils.rnn.pad_sequence(x_seqs, batch_first=True, padding_value=pad_idx)
        y_pad = torch.nn.utils.rnn.pad_sequence(y_seqs, batch_first=True, padding_value=pad_idx)

        yield x_pad, y_pad


# ─────────────────────────────────────────────
# 训练函数
# ─────────────────────────────────────────────
def run_training():
    """训练 LSTM 诗歌语言模型并保存权重。"""
    # 1. 加载数据
    poems_vector, word_to_int, vocabularies = process_poems('./poems.txt')
    vocab_size = len(word_to_int) + 1   # +1 为 padding token 预留
    print(f"词表大小: {vocab_size}，诗歌数量: {len(poems_vector)}")

    # 2. 构建模型
    torch.manual_seed(42)
    embedding = rnn_module.word_embedding(vocab_size, EMBED_DIM)
    model     = rnn_module.RNN_model(vocab_size, embedding, EMBED_DIM, HIDDEN_DIM)

    # 3. 检测设备（优先 GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"使用设备: {device}")

    # 4. 优化器 + 损失函数
    # Adam 通常比 RMSprop 在 NLP 任务上收敛更稳定
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # ignore_index：对 padding 位置不计算损失
    pad_idx   = word_to_int.get(' ', 0)
    loss_fn   = torch.nn.NLLLoss(ignore_index=pad_idx)

    # 5. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for x_batch, y_batch in generate_batch(poems_vector, word_to_int, BATCH_SIZE):
            # 将数据移到目标设备
            x_batch = x_batch.to(device)   # (batch, seq_len)
            y_batch = y_batch.to(device)   # (batch, seq_len)

            # 前向传播：输出 (batch*seq_len, vocab_size) 的对数概率
            log_probs = model(x_batch)

            # 目标展平为 (batch*seq_len,)
            targets = y_batch.contiguous().view(-1)

            # 计算损失
            loss = loss_fn(log_probs, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            # 修复：使用 clip_grad_norm_（带下划线），原版使用的已废弃
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch:02d} | 平均 Loss: {avg_loss:.4f}")

        # 每5个epoch保存一次
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), './poem_generator_rnn_improved')
            print("  ✓ 模型已保存")

    torch.save(model.state_dict(), './poem_generator_rnn_improved')
    print("训练完成，模型已保存至 ./poem_generator_rnn_improved")


# ─────────────────────────────────────────────
# 诗歌生成函数
# ─────────────────────────────────────────────
def gen_poem(begin_word: str, max_length: int = 60) -> str:
    """
    以 begin_word 为起始字，逐步生成一首诗。

    修复点：原版每步都把历史序列全部重新喂入模型，效率低且与 LSTM
    的 stateful 设计矛盾。正确做法是维护隐藏状态 (h, c)，每步只输入
    一个字，依靠隐藏状态传递历史信息。

    参数
    ----
    begin_word : 诗的起始字
    max_length : 生成的最大字数（防止死循环）

    返回
    ----
    生成的诗（字符串）
    """
    poems_vector, word_int_map, vocabularies = process_poems('./poems.txt')
    vocab_size = len(word_int_map) + 1

    # 重建模型并加载权重
    embedding = rnn_module.word_embedding(vocab_size, EMBED_DIM)
    model     = rnn_module.RNN_model(vocab_size, embedding, EMBED_DIM, HIDDEN_DIM)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('./poem_generator_rnn_improved', map_location=device))
    model.to(device)
    model.eval()

    # 初始化隐藏状态
    hidden = model.init_hidden(batch_size=1)
    hidden = (hidden[0].to(device), hidden[1].to(device))

    # 从起始标记开始，将隐藏状态"预热"到 begin_word
    for ch in (START_TOKEN + begin_word[:-1]):
        if ch not in word_int_map:
            continue
        token = torch.tensor([[word_int_map[ch]]], dtype=torch.long, device=device)
        _, hidden = model.generate_step(token, hidden)

    # 逐步生成
    poem  = begin_word
    word  = begin_word[-1]          # 当前字
    with torch.no_grad():
        for _ in range(max_length):
            if word not in word_int_map:
                break
            token = torch.tensor([[word_int_map[word]]], dtype=torch.long, device=device)
            next_token, hidden = model.generate_step(token, hidden)
            word = vocabularies[next_token.item()]
            if word == END_TOKEN:
                break
            poem += word

    return poem


# ─────────────────────────────────────────────
# 格式化打印
# ─────────────────────────────────────────────
def pretty_print_poem(poem: str):
    """将生成的诗按句分行打印。"""
    # 如果诗本身含有标点，可以按标点分行
    # 这里简单按7字或5字分组
    print("=" * 40)
    print(poem)
    print("=" * 40)


# ─────────────────────────────────────────────
# 主程序入口
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # ── 训练阶段（首次运行时执行，后续可注释掉）──
    run_training()

    # ── 生成阶段 ─────────────────────────────────
    print("\n========== 诗歌生成 ==========")
    for begin in ["日", "红", "山", "夜", "湖", "海", "月"]:
        poem = gen_poem(begin)
        print(f"\n【{begin}】")
        pretty_print_poem(poem)