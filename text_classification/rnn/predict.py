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
from paddlenlp.data import Tuple, Pad, Stack, Vocab, JiebaTokenizer

import paddle.nn.functional as F

from model import BoWModel
from utils import preprocess_prediction_data

parser = argparse.ArgumentParser(__doc__)

parser.add_argument("--device", choices=["cpu", "gpu", "xpu"], default="cpu", help="默认选择的设备")
parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
parser.add_argument("--vocab_path", type=str, default="outputs/vocab.json", help="字典表的路径")
parser.add_argument("--network",
                    choices=['bow', 'lstm', 'bilstm', 'gru', 'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn'],
                    default="bow",
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


# 主逻辑函数
if __name__ == '__main__':
    # 默认先设置设备
    paddle.set_device(args.device)

    # 加载词表
    vocab = Vocab.from_json(args.vocab_path)
    # 标签对照表
    label_map = {0: 'negative', 1: 'positive'}

    # 构建网络
    network = args.network.lower()
    # 获取词表长度
    vocab_size = len(vocab)
    # 获取类别长度
    num_classes = len(label_map)
    # 补齐id
    pad_token_id = vocab.to_indices('[PAD]')

    if network == "bow":
        # 针对词带模型
        model = BoWModel(vocab_size=vocab_size, num_classes=num_classes, padding_idx=pad_token_id)
    else:
        raise ValueError(
            "不清楚的网络%s,请输入bow, lstm, bilstm, cnn, gru, bigru, rnn, birnn and bilstm_attn其中的一个" % args.network)

    # 加载模型
    state_dict = paddle.load(args.params_path)
    # 模型填充数据
    model.set_dict(state_dict)

    # 准备数据
    data = [
        '非常不错，服务很好，位于市中心区，交通方便，不过价格也高！',
        '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片',
        '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。',
    ]
    # 转换touken必须和训练的时候一致
    tokenizer = JiebaTokenizer(vocab)

    # 获取一个列表型，内部含有id和句子长度的数据
    examples = preprocess_prediction_data(data, tokenizer)

    # 调用预测，对所有数据进行分类
    results = predict(model, examples, label_map=label_map, batch_size=args.batch_size,
                      pad_token_id=vocab.token_to_idx.get("[PAD]", 0))

    # 打印预测的结果
    for idx, text in enumerate(data):
        print("Data:{} \t label:{}".format(text, results[idx]))
