#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-07 14:23
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    export_model.py
# @Project: rnn
# @Package:
# @Ref:

import argparse

import paddle
from paddlenlp.data import Vocab

from model import BoWModel

parser = argparse.ArgumentParser((__doc__))
parser.add_argument("--vocab_path", type=str,

                    default="outputs/vocab.json", help="词表的路径")
parser.add_argument("--network", choices=['bow', 'lstm', 'bilstm', 'gru',
                                          'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn'], default="bow", help="选择的神经网络")
parser.add_argument("--device", choices=["cpu", "gpu", "xpu"], help="选择运行的设备")
parser.add_argument("--params_path", type=str,
                    default="checkpoints/final.pdparams", help="训练好的模型地址")
parser.add_argument("--output_path", type=str,
                    default="outputs/static_graph_params", help="静态模型保存的路径")
args = parser.parse_args()


# 主逻辑寒素
def main():
    # 加载词库
    vocab = Vocab.from_json(args.vocab_path)
    # 加载标签映射表
    label_map = {0: "negativate", 1: "positive"}

    # 构建网络
    network = args.network.lower()

    # 获取词表长度
    vocab_size = len(vocab)
    # 获取类比长度
    num_classes = len(label_map)
    # 获取补齐对应的id
    pad_token_id = vocab.to_indices("[PAD]")

    if network == "bow":
        model = BoWModel(vocab_size, num_classes, padding_idx=pad_token_id)
    else:
        raise ValueError(
            "不知道的模型%s, 必须是以下几个模型bow, lstm, bilstm, cnn, gru, bigru, rnn, birnn and bilstm_attn" % args.network)

    # 加载模型参数
    state_dict = paddle.load(args.params_path)
    # 给模型设置对应参数
    model.set_dict(state_dict)
    # 模型验证
    model.eval()

    # 设置输入和的形状及类型
    inputs = [paddle.static.InputSpec(shape=[None, None], dtype='int64')]
    # if args.network in [
    #     "lstm", "bilstm", "gru", "bigru", "rnn", "birnn", "bilstm_attn"
    # ]:
    #     inputs.append(paddle.static.InputSpec(
    #         shape=[None], dtype='int64'))  # 序列长度

    # 给模型指定输入的格式
    model = paddle.jit.to_static(model, input_spec=inputs)
    # 保存模型
    # TODO 模型前项传播这块参数是怎么输入的？
    paddle.jit.save(model, args.output_path)


if __name__ == '__main__':
    main()
