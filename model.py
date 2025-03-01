# -*- coding: utf-8 -*-
import unicodedata
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu

# 设置设备并固定随机种子
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)


# 数据预处理工具函数
def unicode_to_ascii(s):
    """将unicode字符串转换为ASCII字符"""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    """清洗和标准化输入句子"""
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)  # 在标点周围添加空格
    w = re.sub(r'[" "]+', " ", w)  # 合并多个空格
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)  # 保留指定字符
    return w.strip()


# 数据集类
class LangPairDataset(Dataset):
    """加载和处理西班牙语-英语翻译数据集"""
    fpath = Path("./data_spa_en/spa.txt")
    cache_path = Path("./.cache/lang_pair.npy")

    def __init__(self, mode="train"):
        # 加载或生成缓存数据
        if not self.cache_path.exists():
            with open(self.fpath, "r", encoding="utf8") as file:
                lines = [l.split('\t') for l in file.readlines()]
                pairs = [[preprocess_sentence(w) for w in l] for l in lines]
                self.trg, self.src = zip(*[(eng, spa) for spa, eng in pairs])
                np.save(self.cache_path, {"trg": self.trg, "src": self.src})
        else:
            data = np.load(self.cache_path, allow_pickle=True).item()
            self.trg, self.src = data["trg"], data["src"]

        # 划分训练测试集
        split_idx = np.random.choice(["train", "test"], p=[0.9, 0.1], size=len(self.trg))
        self.trg = np.array(self.trg)[split_idx == mode]
        self.src = np.array(self.src)[split_idx == mode]

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

    def __len__(self):
        return len(self.src)


# 词表构建函数
def build_vocab(ds, mode="src"):
    """构建源语言或目标语言的词表"""
    word2idx = {"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, "[EOS]": 3}
    words = " ".join([pair[0 if mode == "src" else 1] for pair in ds]).split()
    counter = Counter(words)
    for word, count in counter.items():
        if count >= 2:  # 仅保留出现2次以上的词
            word2idx[word] = len(word2idx)
    return word2idx, {v: k for k, v in word2idx.items()}


# 初始化词表
src_word2idx, src_idx2word = build_vocab(LangPairDataset("train"), "src")
trg_word2idx, trg_idx2word = build_vocab(LangPairDataset("train"), "trg")


# 数据编码解码类
class Tokenizer:
    """实现文本与索引序列的转换"""

    def __init__(self, word2idx, max_len=50):
        self.word2idx = word2idx
        self.pad, self.bos, self.eos = 0, 1, 3

    def encode(self, texts, add_special=True):
        """将文本列表编码为填充后的索引矩阵"""
        max_len = max(len(t.split()) for t in texts) + 2 if add_special else 0
        batch = []
        for text in texts:
            indices = [self.word2idx.get(w, 2) for w in text.split()]
            if add_special:
                indices = [self.bos] + indices + [self.eos]
            batch.append(indices + [self.pad] * (max_len - len(indices)))
        return torch.tensor(batch)

    def decode(self, indices):
        """将索引矩阵解码回文本"""
        return [" ".join(self.idx2word.get(i, "[UNK]") for i in indices if i not in [0, 1, 3])]
        # 模型组件


class Encoder(nn.Module):
    """编码器：嵌入层+GRU"""

    def __init__(self, vocab_size, emb_dim=256, hid_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Attention(nn.Module):
    """Bahdanau注意力机制"""

    def __init__(self, hid_dim):
        super().__init__()
        self.W = nn.Linear(hid_dim, hid_dim)
        self.V = nn.Linear(hid_dim, 1)

    def forward(self, query, keys):
        energy = torch.tanh(self.W(keys) + query.unsqueeze(1))
        attention = F.softmax(self.V(energy), dim=1)
        return torch.sum(attention * keys, dim=1)


class Decoder(nn.Module):
    """解码器：注意力+GRU+全连接"""

    def __init__(self, vocab_size, emb_dim=256, hid_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = Attention(hid_dim)
        self.gru = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x, hidden, encoder_out):
        context = self.attention(hidden[-1], encoder_out)
        embedded = self.embedding(x)
        output, hidden = self.gru(torch.cat([embedded, context.unsqueeze(1)], dim=2), hidden)
        return self.fc(output.squeeze(1)), hidden


# 完整模型
class Seq2Seq(nn.Module):
    """带注意力的seq2seq模型"""

    def __init__(self, src_vocab, trg_vocab):
        super().__init__()
        self.encoder = Encoder(len(src_vocab))
        self.decoder = Decoder(len(trg_vocab))
        self.trg_vocab = trg_vocab

    def forward(self, src, trg):
        enc_out, hidden = self.encoder(src)
        output = []
        for i in range(trg.size(1)):
            dec_out, hidden = self.decoder(trg[:, i], hidden, enc_out)
            output.append(dec_out)
        return torch.stack(output, dim=1)

    def predict(self, src, max_len=50):
        """推理方法"""
        with torch.no_grad():
            enc_out, hidden = self.encoder(src)
            trg = torch.tensor([[self.trg_vocab["[BOS]"]]]).to(device)
            output = []
            for _ in range(max_len):
                dec_out, hidden = self.decoder(trg, hidden, enc_out)
                trg = dec_out.argmax(1).unsqueeze(1)
                output.append(trg.item())
                if trg.item() == self.trg_vocab["[EOS]"]:
                    break
        return output


# 训练准备
def collate_fn(batch):
    """数据加载的批处理函数"""
    src_batch, trg_batch = zip(*batch)
    src_tensor = Tokenizer(src_word2idx).encode(src_batch)
    trg_tensor = Tokenizer(trg_word2idx).encode(trg_batch)
    return src_tensor.to(device), trg_tensor.to(device)


# 初始化模型和优化器
model = Seq2Seq(src_word2idx, trg_word2idx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)


# 训练循环
def train(model, epochs=10, batch_size=32):
    train_loader = DataLoader(LangPairDataset("train"), batch_size, shuffle=True, collate_fn=collate_fn)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, trg in train_loader:
            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            loss = loss_fn(output.reshape(-1, len(trg_word2idx)), trg[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")


# 评估函数
def evaluate(model, test_set):
    """计算测试集的BLEU分数"""
    scores = []
    for src, ref in test_set:
        src_tensor = Tokenizer(src_word2idx).encode([src]).to(device)
        pred = model.predict(src_tensor)
        candidate = Tokenizer(trg_word2idx).decode([pred])[0]
        scores.append(sentence_bleu([ref.split()], candidate.split(), weights=(0.25, 0.25, 0.25, 0.25)))
    return np.mean(scores)


# 主程序
if __name__ == "__main__":
    train(model, epochs=20)
    test_set = LangPairDataset("test")
    print(f"BLEU Score: {evaluate(model, test_set):.4f}")