#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-07 14:25
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    model.py
# @Project: rnn
# @Package: 
# @Ref:

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as nlp


class BoModel(nn.Layer):
    def __init__(self, vocab_size, num_classes, emb_dim=128, padding_idx=0, hidden_size=128, fc_hidden_size=96):
        """
        词带模型，输出输出维度一样，输入一个序列的，输出的事这个序列加和平均
        :param vocab_size: 词表大小
        :param num_classes: 待分类个数
        :param emb_dim: 词表维度
        :param padding_idx: 补齐文字的id
        :param hidden_size: 隐藏层大小
        :param fc_hidden_size: 全连接层隐藏层大小
        """
        super(BoModel, self).__init__()

        # 创建模型的词表embedding
        self.embedder = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        # 创建词带模型，输入的事词矩阵的维度大小
        self.bow_encoder = nlp.seq2vec.BoWEncoder(emb_dim=emb_dim)
        # 词带模型后面添加一个词带模型, 输入的大小是词带模型输出的维度，输出维度是隐藏层的大小默认128
        self.fc1 = nn.Linear(self.bow_encoder.get_output_dim(), hidden_size)
        # 输入的维度是上一个全连接层的输出维度，
        self.fc2 = nn.Linear(hidden_size, fc_hidden_size)
        # 输出的即为业务维度96
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        """

        :param text:
        :param seq_len:
        :return:
        """
        # 形状：(batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)  # 直接将文字转换成embedding矩阵

        # 形状：(batch_size, hidden_size)
        summed = self.bow_encoder(embedded_text)  # 将embedding输入到词带模型中
        # 对输出的张量进行激活,tanh也是一种非常常见的激活函数。与sigmoid相比，它的输出均值是0，使得其收敛速度要比sigmoid快，
        # 减少迭代次数。然而，从途中可以看出，tanh一样具有软饱和性，从而造成梯度消失
        encoded_text = paddle.tanh(summed)

        # 形状：(batch_size, hidden_size)
        fc1_out = paddle.tanh(self.fc1(encoded_text))
        # 形状：(batch_size,fc_hidden_size)
        fc2_out = paddle.tanh(self.fc2(fc1_out))
        # 形状：（batch_size, num_classes）
        logits = self.output_layer(fc2_out)
        return logits
