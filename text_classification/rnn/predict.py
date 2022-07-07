#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-07 14:31
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    predict.py
# @Project: rnn
# @Package: 
# @Ref:


import argparse

import paddle
from paddlenlp.data import Tuple, Pad, Stack

import paddle.nn.functional as F

parser = argparse.ArgumentParser(__doc__)

parser.add_argument("--device", choices=["cpu", "gpu", "xpu"], default="cpu", help="默认选择的设备")
parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
parser.add_argument("--vocab_path", type=str, default="output/vocab.json", help="字典表的路径")
parser.add_argument("--network",
                    choices=['bow', 'lstm', 'bilstm', 'gru', 'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn'],
                    help="待选择的网络")

parser.add_argument("--params_path", type=str, default="checkpoints/final.pdparams", help="训练好的模型的地址")
args = parser.parse_args()


def predict(model, data, label_map, batch_size=1, pad_token_id=0):
    """
    预测的部分
    :param model: 预测所使用的模型
    :param data: 预测的数据
    :param label_map: 标签的映射
    :param batch_size: 与测试后的批次大小
    :param pad_token_id: 补全部分的PAD对应的id
    :return: 返回所有批次结果
    """
    # 二维数据，内层是一个一个数据，按照batch进行切片
    batches = [
        data[idx:idx + batch_size] for idx in range(0, len(data), batch_size)
    ]

    # TODO 需要仔细研究
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=pad_token_id),  # 输入文字的id
        Stack(dtype='int64'),  # 序列的长度
    ): [data for data in fn(samples)]

    # 多条结果存取容器
    results = []
    model.eval()
    for batch in batches:
        # 获取每一个batch数据
        texts, seq_lens = batchify_fn(batch)
        # 对每一个batch数据，即多句话转成张量
        texts = paddle.to_tensor(texts)
        # 对每句话的长度也转层张量
        seq_lens = paddle.to_tensor(seq_lens)
        # 把一个批次数据送入模型，获取最后一层全连接层的维度数组
        logits = model(texts, seq_lens)

        # 根据softmax，算出每一句话，对应所有类型的概率,轴0代表每句话，轴1对应每句话的所有分类概率
        probs = F.softmax(logits, axis=1)
        # 获取每一句话概率最大值的索引，转换成ndarray类型
        idx = paddle.argmax(probs, axis=1).numpy()
        # 将结果转成列表
        idx = idx.tolist()
        # 根据id查询对应的标签，方便其他人查看
        labels = [label_map[i] for i in idx]
        # 对一个批次的结果拼接
        results.extend(labels)
    return results
