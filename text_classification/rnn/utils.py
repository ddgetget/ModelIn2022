#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-07 14:31
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    utils.py
# @Project: rnn
# @Package: 
# @Ref:
# 一种高性能的字典
from collections import defaultdict
from paddlenlp import Taskflow

word_segmenter = Taskflow("word_segmentation", mode="fast")


def build_vocab(texts, stopwords=[], num_words=None, min_freq=10, unk_token="[UNK]", pad_token="[PAD]"):
    """
    构建词表索引
    :param texts: 构建的索引的文字
    :param stopwords: 停用词
    :param num_words: 词库最大限制
    :param min_freq: 过滤词频小于当前值的
    :param unk_token: 对于未知类型的token-UNK
    :param pad_token: 对于需要补齐的token-PAD
    :return: 整个语料的索引字典
    """
    pass
    # TODO 梳理构建词表的逻辑
    # 声明一个wordcount的字典，key为单词，value为当前值的频数
    word_counts = defaultdict(int)
    # 遍历所有的文字，填充这个词表字典
    for text in texts:
        # 如果字典里没有这个词语继续进行
        if not text:
            continue

        #
        # 对当前的句子进行分词
        for word in word_segmenter(text):
            # 剔除在停用词里面的
            if word in stopwords:
                continue
            # 统一每一个词的频数
            word_counts[word] += 1

    # 构建一个word列表，存取word,count元组
    wcounts = []
    # 遍历字典里每一对键值对
    for word, count in word_counts.items():
        # 剔除频率较低的词语
        if count < min_freq:
            continue

        # 添加真正的word cound元组
        wcounts.append((word, count))

    # 针对word count进行按频数降序排列
    wcounts.sort(key=lambda x: x[1], reverse=True)

    # 针对词表最大值限制为None，或者wordcount大于词表最大限制去除两个占位的
    if num_words is not None and len(wcounts) > (num_words - 2):
        # 删除大于词表限制的词语,后面的词语频数较小
        wcounts = wcounts[:(num_words - 2)]

    # 添加两个特殊的暗号标记
    sorted_voc = [pad_token, unk_token]
    # 拼接wordcount到有暗号的列表里去
    sorted_voc.extend(wc[0] for wc in wcounts)

    # 按照含有暗号的列表，对应的索引进行构建词索引, 后半截获得一个range （0， total_word_count）
    word_index = dict(zip(sorted_voc, list(range(len(sorted_voc)))))

    # 返回一个含有单词索引的，且经过逻辑处理的字典语料库
    return word_index
