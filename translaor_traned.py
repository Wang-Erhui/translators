# -*- coding: utf-8 -*-
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from model import Seq2Seq, Tokenizer, preprocess_sentence, LangPairDataset, src_word2idx, trg_word2idx, device


class Translator:
    """加载训练好的翻译模型进行推理"""

    def __init__(self, ckpt_path="best.ckpt"):
        # 初始化与训练时相同的模型结构
        self.model = Seq2Seq(src_word2idx, trg_word2idx).to(device)

        # 加载训练好的权重
        checkpoint = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()  # 设置为评估模式

        # 初始化tokenizers（必须与训练时相同）
        self.src_tokenizer = Tokenizer(src_word2idx)
        self.trg_tokenizer = Tokenizer(trg_word2idx)

    def preprocess(self, sentence):
        """输入句子预处理（与训练时完全一致）"""
        return preprocess_sentence(sentence)

    def translate(self, sentence, max_length=50):
        """执行单句翻译"""
        with torch.no_grad():
            # 预处理和编码
            cleaned = self.preprocess(sentence)
            src_tensor = self.src_tokenizer.encode([cleaned.split()], add_special=True).to(device)

            # 模型推理
            pred_ids = self.model.predict(src_tensor, max_len=max_length)

            # 解码结果（过滤特殊标记）
            return self.trg_tokenizer.decode([pred_ids])[0]

    def evaluate_bleu(self, test_samples=100):
        """评估模型在测试集上的表现"""
        scores = []
        test_set = LangPairDataset("test")[:test_samples]

        for src, ref in test_set:
            candidate = self.translate(src).split()
            reference = [ref.split()]
            scores.append(sentence_bleu(reference, candidate))

        return np.mean(scores)


if __name__ == "__main__":
    # 初始化翻译器
    translator = Translator()

    # 示例翻译
    test_sentences = [
        "¿Qué hora es?",  # What time is it?
        "Hace buen tiempo hoy.",  # The weather is nice today.
        "¿Puedo ayudarte?",  # Can I help you?
    ]

    for sent in test_sentences:
        translation = translator.translate(sent)
        print(f"西班牙语: {sent}")
        print(f"英语翻译: {translation}\n")

    # 评估模型表现
    bleu_score = translator.evaluate_bleu(test_samples=500)
    print(f"模型在500个测试样本上的BLEU-4分数: {bleu_score:.4f}")