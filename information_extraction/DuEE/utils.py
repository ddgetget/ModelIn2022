#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-11 22:34
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    utils.py
# @Project: DuEE
# @Package: 
# @Ref:


import hashlib


def cal_md5(str):
    """获取MD5码"""
    str = str.decode("utf-8", "ignore").encode("utf-8", "ignore")
    return hashlib.md5(str).hexdigest()


def read_by_lines(path):
    """按行读取数据"""
    result = list()
    with open(path, "r", encoding="utf8") as infile:
        # 遍历每一行数据
        for line in infile:
            # 每一行数据去除两边空格，拼接
            result.append(line.strip())
    # 返回数据
    return result


def write_by_lines(path, data):
    """写数据到本地"""
    with open(path, "w", encoding="utf8") as outfile:
        # 按行写到本地
        [outfile.write(d + "\n") for d in data]


def text_to_sents(text):
    """文本转句子"""
    # 分割字符
    deliniter_symbols = [u"。", u"？", u"！"]
    # 按行切割
    paragraphs = text.split("\n")
    ret = []
    # 遍历段落
    for para in paragraphs:
        # 每一段是u的重新循环
        if para == u"":
            continue
        sents = [u""]
        # 遍历段落里每一个字符，根据一句话终止符份额高
        for s in para:
            # 拼接字符串 u"apple"
            sents[-1] += s
            # 判断当前字符是否在分割字符里面，也就是说是不是一句话
            if s in deliniter_symbols:
                # 如果在，后面补一个空字符
                sents.append(u"")
        if sents[-1] == u"":
            # 判断一句话最后是 u""结尾，那么截全掉最后那个空格
            sents = sents[:-1]
        # 把处理过的数据拼接到结果集上
        ret.extend(sents)
    return ret


def load_dict(dict_path):
    """加载词典"""
    # 分配一个装字典的空间
    vocab = {}
    # 按行读取
    for line in open(dict_path, 'r', encoding='utf-8'):
        # 获取key和value，他们之间是用TAB分割的
        value, key = line.strip('\n').split('\t')
        # 装入字典
        # {"B-股东减持":"4"}
        vocab[key] = int(value)
    return vocab


def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    if len(text) != len(labels):
        # 韩文回导致label 比 text要长
        labels = labels[:len(text)]
    for i, label in enumerate(labels):
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret


if __name__ == "__main__":
    s = "xxdedewd"
    print(cal_md5(s.encode("utf-8")))
