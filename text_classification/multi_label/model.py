#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-11 21:58
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    model.py
# @Project: multi_label
# @Package: 
# @Ref:
import paddle.nn as nn


class MultiLabelClassifier(nn.Layer):

    def __init__(self, pretrained_model, num_labels=2, dropout=None):
        super(MultiLabelClassifier, self).__init__()
        self.ptm = pretrained_model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ptm.
                                  config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], num_labels)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, pooled_output = self.ptm(input_ids,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
