#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-11 22:29
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    export_model.py
# @Project: pretrained_models
# @Package: 
# @Ref:


import argparse
import os

import paddle
import paddlenlp as ppnlp

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, default='./checkpoint/model_900/model_state.pdparams',
                    help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./export',
                    help="The path of model parameter in static graph to be saved.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    # The number of labels should be in accordance with the training dataset.
    label_map = {0: 'negative', 1: 'positive'}
    model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(
        "ernie-1.0", num_classes=len(label_map))

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    model.eval()

    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None],
                                    dtype="int64")  # segment_ids
        ])
    # Save in static graph model.
    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
