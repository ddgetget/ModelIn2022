#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-11 22:28
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    utils.py
# @Project: pretrained_models
# @Package: 
# @Ref:


import numpy as np


def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False,
                    is_pair=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed
    to be used in a sequence-pair classification task.

    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``

    It returns the first portion of the mask (0's).


    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """

    if is_pair:
        text = example["text_a"]
        text_pair = example["text_b"]
    else:
        text = example["text"]
        text_pair = None
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair,
                               max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if is_test:
        return input_ids, token_type_ids
    label = np.array([example["label"]], dtype="int64")
    return input_ids, token_type_ids, label
