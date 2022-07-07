#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-07 14:31
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    train.py
# @Project: rnn
# @Package: 
# @Ref:

# 系统包，做一些系统级别的命令
import os

# 接受终端，并解析给程序使用，是程序和开发有人员的交通枢纽
import argparse
# 典型的，函数在执行时，要带上所有必要的参数进行调用。然后，有时参数可以在函数被调用之前提前获知。这种情况下，一个函数有一个或多个参数预先
# 就能用上，以便函数能用更少的参数进行调用。
from functools import partial
# 随机数工具包
import random

# 科学计数宝
import numpy as np

# 百度神经网络计算框架
import paddle
# 加载数据集
from paddlenlp.datasets import load_dataset

# paddlenlp领域的数据处理常用方法汇集
from paddlenlp.data import Vocab, JiebaTokenizer

# 使用工具集里面咋们自己的构建方法
from model import BoModel
from utils import build_vocab,convert_example

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=15, help="训练的总次数")
parser.add_argument("--device", choices=['cpu', 'gpu', 'xpu', 'npu'], default='cpu', help="选择训练的机器设备")
parser.add_argument("--lr", type=float, default=5e-5, help="训练的学习率")
parser.add_argument("--save_dir", type=str, default="checkpoints/", help="模型保存的默认路径")
parser.add_argument("--batch_size", type=int, default=64, help="默认训练批次大小")
parser.add_argument("--vocab_path", type=str, default="output/vocab.json", help="默认保存词典的位置")
parser.add_argument("--netwoprk",
                    choices=['bow', 'lstm', 'bilstm', 'gru', 'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn'],
                    default='bow', help="选择神经网络,默认词带模型bow")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="初始化模型加载路径")

args = parser.parse_args()


def set_seed(seed=1000):
    """
    设置全局随机种子
    :param seed: 随机种子
    :return:
    """

    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def create_dataloader(dataset, trans_fn=None, mode='train', batch_size=1, batchify_fn=None):
    """
    构建数据集
    :param dataset: 数据集路径
    :param trans_fn: 将文字数据转换成输入可用的数字id
    :param mode: 数据集模式(训练集，测试集，验证集)
    :param batch_size: 数据每个批次大小
    :param batchify_fn: TODO 补充说明
    :return: paddle.io.DataLoader
    """
    # 针对训练集，因为有标签一列，需要特殊处理，转换成文字对应的id
    if trans_fn:
        dataset = dataset.map(trans_fn)

    # 为了确保数据集的分布均衡，做了打乱操作，当然对于训练集有用，测试集和验证集因为不需要学习新的规律，所以不需要
    shuffle = True if mode == "train" else False

    if mode == "train":
        # 对训练集进行采样，具体说明可以参考：https://ew6tx9bj6e.feishu.cn/docx/doxcnB0u7eOZ8Q6WUTa2rgym94W#doxcnMe4A6qwkAqy6kZvLF0xmtg
        # 分布式批采样器加载数据的一个子集。每个进程可以传递给DataLoader一个DistributedBatchSampler的实例，每个进程加载原始数据的一个子集。
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        # 普通数据批次采样
        sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    # 获取数据加载器，跟上面的采样器一样，也是一个迭代器，方便大数据量的运行，节省内存
    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, collate_fn=batchify_fn)
    return dataloader


# 主逻辑函数
if __name__ == '__main__':
    # 设置默认机器选型，这里默认是CPU
    paddle.set_device(args.device)

    # 加载数据
    train_ds, dev_ds = load_dataset("chnsenticorp", splits=['train', 'dev'])

    # 构造一个存数据的容器
    texts = []
    # 所有训练集拼接到列表容器
    for data in train_ds:
        texts.append(data['text'])
    # 所有验证集数据拼接到容器
    for data in dev_ds:
        texts.append(data['text'])

    # 准备一些停用词，这些词词频较高，对结果严重扰乱
    stopwords = set(["的", "吗", "吧", "呀", "呜", "呢", "呗"])

    # 构建词表
    word2idx = build_vocab(texts, stopwords, min_freq=5, unk_token="[UNK]", pad_token="[PAD]")

    vocab = Vocab.from_dict(word2idx, unk_token='[UNK]', pad_token="[PAD]")
    # 保存词典库

    # 框架工具自身不能创建文件夹，这里需要自己手动创建
    if not os.path.exists("output"):
        os.mkdir("output")
    vocab.to_json(args.vocab_path)

    # 组网部分,防止用户输出大写的网络明湖曾
    network = args.network.lower()
    # 计算当前有效词表的大小
    vocab_size = len(vocab)
    # 计算训练集的不同标签个数
    num_classes = len(train_ds.label_list)

    # 获取PDA这个token的索引，气=其实之前我们已经知道了，在0号位置
    pad_token_id = vocab.to_indices('[PAD]')

    if network == "bow":
        # 加载BOW模型
        model = BoModel(vocab_size=vocab_size, num_classes=num_classes, padding_idx=pad_token_id)
    else:
        raise ValueError("不知道的模型网络:%s，你只能选择bow, lstm, bilstm, cnn, gru, bigru, rnn, birnn and bilstm_attn" % network)

    # 读取数据并弄成生成器型的mini-batchs
    tokenizer = JiebaTokenizer(vocab)
    trans_fn = partial(convert_example, tokenizer=tokenizer, is_test=False)
