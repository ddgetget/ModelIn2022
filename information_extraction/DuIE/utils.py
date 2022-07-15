#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-12 21:38
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    utils.py
# @Project: DuIE
# @Package: 
# @Ref:


import codecs
import json
import os
import re
import zipfile

import numpy as np


def find_entity(text_raw, id_, predictions, tok_to_orig_start_index,
                tok_to_orig_end_index):
    """
    获取实体
    :param text_raw: 原始文本
    :param id_: 对应id
    :param predictions: 预测结果
    :param tok_to_orig_start_index: 实体其实索引
    :param tok_to_orig_end_index: 实体结束索引
    :return:
    """
    # 手机实体结果列表
    entity_list = []
    # 遍历预测结果
    for i in range(len(predictions)):
        # 针对id在预测结果内
        if [id_] in predictions[i]:
            # 设置标记为j
            j = 0

            while i + j + 1 < len(predictions):
                # 针对标签是1的，继续贪心寻找下一个
                if [1] in predictions[i + j + 1]:
                    # 获取最大的可用的下标索引
                    j += 1
                else:
                    break
            # 获取一个实体
            entity = ''.join(
                text_raw[tok_to_orig_start_index[i]:tok_to_orig_end_index[i + j] + 1])
            # 拼接一个结果
            entity_list.append(entity)
    # 去重，然后再转换成列表
    return list(set(entity_list))


def decoding(example_batch, id2spo, logits_batch, seq_len_batch,
             tok_to_orig_start_index_batch, tok_to_orig_end_index_batch):
    """
    model output logits -> formatted spo (as in data set file)
    """
    formatted_outputs = []
    for (i, (example, logits, seq_len, tok_to_orig_start_index, tok_to_orig_end_index)) in \
            enumerate(zip(example_batch, logits_batch, seq_len_batch, tok_to_orig_start_index_batch,
                          tok_to_orig_end_index_batch)):

        logits = logits[1:seq_len +
                          1]  # slice between [CLS] and [SEP] to get valid logits
        logits[logits >= 0.5] = 1
        logits[logits < 0.5] = 0
        tok_to_orig_start_index = tok_to_orig_start_index[1:seq_len + 1]
        tok_to_orig_end_index = tok_to_orig_end_index[1:seq_len + 1]
        predictions = []
        for token in logits:
            predictions.append(np.argwhere(token == 1).tolist())

        # format predictions into example-style output
        formatted_instance = {}
        text_raw = example['text']
        complex_relation_label = [8, 10, 26, 32, 46]
        complex_relation_affi_label = [9, 11, 27, 28, 29, 33, 47]

        # flatten predictions then retrival all valid subject id
        flatten_predictions = []
        for layer_1 in predictions:
            for layer_2 in layer_1:
                flatten_predictions.append(layer_2[0])
        subject_id_list = []
        for cls_label in list(set(flatten_predictions)):
            if 1 < cls_label <= 56 and (cls_label + 55) in flatten_predictions:
                subject_id_list.append(cls_label)
        subject_id_list = list(set(subject_id_list))

        # fetch all valid spo by subject id
        spo_list = []
        for id_ in subject_id_list:
            if id_ in complex_relation_affi_label:
                continue  # do this in the next "else" branch
            if id_ not in complex_relation_label:
                subjects = find_entity(text_raw, id_, predictions,
                                       tok_to_orig_start_index,
                                       tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions,
                                      tok_to_orig_start_index,
                                      tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        spo_list.append({
                            "predicate":
                                id2spo['predicate'][id_],
                            "object_type": {
                                '@value': id2spo['object_type'][id_]
                            },
                            'subject_type':
                                id2spo['subject_type'][id_],
                            "object": {
                                '@value': object_
                            },
                            "subject":
                                subject_
                        })
            else:
                #  traverse all complex relation and look through their corresponding affiliated objects
                subjects = find_entity(text_raw, id_, predictions,
                                       tok_to_orig_start_index,
                                       tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions,
                                      tok_to_orig_start_index,
                                      tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        object_dict = {'@value': object_}
                        object_type_dict = {
                            '@value': id2spo['object_type'][id_].split('_')[0]
                        }
                        if id_ in [8, 10, 32, 46
                                   ] and id_ + 1 in subject_id_list:
                            id_affi = id_ + 1
                            object_dict[id2spo['object_type'][id_affi].split(
                                '_')[1]] = find_entity(text_raw, id_affi + 55,
                                                       predictions,
                                                       tok_to_orig_start_index,
                                                       tok_to_orig_end_index)[0]
                            object_type_dict[
                                id2spo['object_type'][id_affi].split('_')
                                [1]] = id2spo['object_type'][id_affi].split(
                                '_')[0]
                        elif id_ == 26:
                            for id_affi in [27, 28, 29]:
                                if id_affi in subject_id_list:
                                    object_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                        find_entity(text_raw, id_affi + 55, predictions, tok_to_orig_start_index,
                                                    tok_to_orig_end_index)[0]
                                    object_type_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                        id2spo['object_type'][id_affi].split('_')[0]
                        spo_list.append({
                            "predicate":
                                id2spo['predicate'][id_],
                            "object_type":
                                object_type_dict,
                            "subject_type":
                                id2spo['subject_type'][id_],
                            "object":
                                object_dict,
                            "subject":
                                subject_
                        })

        formatted_instance['text'] = example['text']
        formatted_instance['spo_list'] = spo_list
        formatted_outputs.append(formatted_instance)
    return formatted_outputs


def write_prediction_results(formatted_outputs, file_path):
    """
    预测结果写到本地
    :param formatted_outputs: 预测结果
    :param file_path: 写入文件路径
    :return:
    """
    # 打开文件
    with codecs.open(file_path, 'w', 'utf-8') as f:
        # 年里每一行结果
        for formatted_instance in formatted_outputs:
            # 将结果转换成字符串
            json_str = json.dumps(formatted_instance, ensure_ascii=False)
            # 写入本地
            f.write(json_str)
            f.write('\n')
        # zip压缩文件路径
        zipfile_path = file_path + '.zip'
        # 设置zip文件参数
        f = zipfile.ZipFile(zipfile_path, 'w', zipfile.ZIP_DEFLATED)
        # zip文件写到本地
        f.write(file_path)
    # 返回压缩文件路径
    return zipfile_path


def get_precision_recall_f1(golden_file, predict_file):
    """
    获取准确率，召回率，和F1-Score
    :param golden_file:
    :param predict_file:
    :return:
    """
    # 运行当前脚本
    r = os.popen(
        'python3 ./re_official_evaluation.py --golden_file={} --predict_file={}'
        .format(golden_file, predict_file))
    # 收集结果
    result = r.read()
    # 关闭管道
    r.close()
    # 获取准确率
    precision = float(
        re.search(
            "\"precision\", \"value\":.*?}",
            result).group(0).lstrip("\"precision\", \"value\":").rstrip("}"))  # 正则抽取日志结果
    # 获取召回率
    recall = float(
        re.search("\"recall\", \"value\":.*?}",
                  result).group(0).lstrip("\"recall\", \"value\":").rstrip("}"))  # 正则抽取日志结果
    # 获取F1-score
    f1 = float(
        re.search(
            "\"f1-score\", \"value\":.*?}",
            result).group(0).lstrip("\"f1-score\", \"value\":").rstrip("}"))  # 正则抽取日志结果
    # 返回结果
    return precision, recall, f1
