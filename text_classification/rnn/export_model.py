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
from paddlenlp.data import Vocab

parser = argparse.ArgumentParser((__doc__))
parser.add_argument("--vocab_path", type=str,

                    default="output/vocab.json", help="词表的路径")
parser.add_argument("--network", choices=['bow', 'lstm', 'bilstm', 'gru',
                    'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn'], help="选择的神经网络")
parser.add_argument("--device",choices=["cpu","gpu","xpu"],help="选择运行的设备")
parser.add_argument("--params_path",type=str,default="checkpoints/final.pdparams",help="训练好的模型地址")
parser.add_argument("--output_path",type=str,default="output/static_graph_params",help="静态模型保存的路径")
args=parser.parse_args()



# 主逻辑寒素
def main():
    # 加载词库
    vocab=Vocab.from_json(args.vocab_path)
