#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-12 21:42
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    data_loader.py
# @Project: DuIE
# @Package: 
# @Ref:


import collections
import json
import os
from typing import Optional, List, Union, Dict

import numpy as np
import paddle
# 带来了比较好优化类方案，提供的各类方法也足够用，可以在之后的项目里面逐渐使用起来。
from dataclasses import dataclass
# 自然语言处理预训练模型
from paddlenlp.transformers import ErnieTokenizer

# 中文和标点符号抽取处理器
from extract_chinese_and_punct import ChineseAndPunctuationExtractor

# 定义输入特征的结构，类似C++的结构体
InputFeature = collections.namedtuple("InputFeature", [
    "input_ids", "seq_len", "tok_to_orig_start_index", "tok_to_orig_end_index",
    "labels"
])


def parse_label(spo_list, label_map, tokens, tokenizer):
    """
    解析标签
    :param spo_list: 标签㽹
    :param label_map: 标签字典
    :param tokens: 对应的词表token
    :param tokenizer: token解析器
    :return:
    """
    # 2 tags for each predicate + I tag + O tag
    # 计算所有标签的总数，包括B，I，O，公式为：2N+1，标签包含起始标签
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    # 计算tokens词表的长度
    seq_len = len(tokens)
    # initialize tag
    # 初始化标签矩阵，每一个字符串都是num_labels个标签
    labels = [[0] * num_labels for i in range(seq_len)]
    #  find all entities and tag them with corresponding "B"/"I" labels
    # 查找所有的实体，并对其进行标签B/I化

    """
    {
    "text":"吴宗宪遭服务生种族歧视, 他气呛: 我买下美国都行!艺人狄莺与孙鹏18岁的独子孙安佐赴美国读高中，没想到短短不到半年竟闹出校园安全事件被捕，因为美国正处于校园枪击案频传的敏感时机，加上国外种族歧视严重，外界对于孙安佐的情况感到不乐观 吴宗宪今（30）日录影前谈到美国民情，直言国外种族歧视严重，他甚至还被一名墨西哥裔的服务生看不起，让吴宗宪气到喊：「我是吃不起是不是",
    "spo_list":[
        {
            "predicate":"父亲",
            "object_type":{
                "@value":"人物"
            },
            "subject_type":"人物",
            "object":{
                "@value":"孙鹏"
            },
            "subject":"孙安佐"
        },
            ]
    }
    """
    for spo in spo_list:
        # 获取数据里面的所有object
        for spo_object in spo['object'].keys():
            # assign relation label
            # 判断关系是否在标签里面
            if spo['predicate'] in label_map.keys():
                # simple relation
                # 获取关系标签
                label_subject = label_map[spo['predicate']]
                # 主题对应标签id,55的原因是，predicate2id.json链有57个标签，抛去I，O标签，还有55个
                label_object = label_subject + 55
                # 查词典表，获取subject对应的toeken
                subject_tokens = tokenizer._tokenize(spo['subject'])
                # 查找objectd对应真实值的的token
                object_tokens = tokenizer._tokenize(spo['object']['@value'])
            else:
                # complex relation
                label_subject = label_map[spo['predicate'] + '_' + spo_object]
                label_object = label_subject + 55
                # 解析客体的token
                subject_tokens = tokenizer._tokenize(spo['subject'])
                # TODO 暂时不清楚
                object_tokens = tokenizer._tokenize(spo['object'][spo_object])

            # 计算客体token长度
            subject_tokens_len = len(subject_tokens)
            # 计算主题的token长度
            object_tokens_len = len(object_tokens)

            # assign token label
            # there are situations where s entity and o entity might overlap, e.g. xyz established xyz corporation
            # to prevent single token from being labeled into two different entity
            # we tag the longer entity first, then match the shorter entity within the rest text
            # 为了防止标签有重叠，这=这里选择较长的开始计算，然后匹配最短的
            forbidden_index = None
            # 【客体】针对客体长度大于主体长度
            if subject_tokens_len > object_tokens_len:
                # 针对客体每一个token，先处理较长字符串
                for index in range(seq_len - subject_tokens_len + 1):
                    # 截取客体长度各字符，判断是否和客体token长度相同，理论上应该是相等的吧
                    if tokens[index:index +
                              subject_tokens_len] == subject_tokens:
                        # 标记第index个字符的客体的位1，之前默认是0
                        labels[index][label_subject] = 1
                        # 针对当前位置之后的subject_tokens_len长度的标签都修改为1
                        for i in range(subject_tokens_len - 1):
                            labels[index + i + 1][1] = 1
                        # 记录最后一个修改过的字符的位置
                        forbidden_index = index
                        break
                # 针对主题每一个token
                for index in range(seq_len - object_tokens_len + 1):
                    # 截取object_tokens_len长度各字符，理论上这是等于的
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        # 针对之前是灭有客体的，也就是上一步没有客体的
                        if forbidden_index is None:
                            # 修改字符赌赢的标签
                            labels[index][label_object] = 1
                            # 以此之后object_tokens_len长度的标签都修改成1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            # 然后表现处理完毕
                            break
                        # check if labeled already
                        # 针对索引在客体最后一个标签之前，或者索引在客体最后一个标签之后，也就是所主题的最后一个字符，肯定在客体的前面片段，或者客体片段之后。
                        elif index < forbidden_index or index >= forbidden_index + len(
                                subject_tokens):
                            # 这里的主体标签才是有效的，否则会有交叉，不大对劲，对应基础算法，字符串查找，
                            labels[index][label_object] = 1
                            # 同理，对之后一段区间的字符串修改标记
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break

            else:
                # 【主体】处理主体长度大于客体长度的，主体长
                for index in range(seq_len - object_tokens_len + 1):
                    # 判断当前所选字符是否为这正的客体字符
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        # 标签第一个位置的标签为1
                        labels[index][label_object] = 1
                        # 接着对后续的object_tokens_len也设置成1
                        for i in range(object_tokens_len - 1):
                            labels[index + i + 1][1] = 1
                        # 做一个标记，说明主题已经标记了
                        forbidden_index = index
                        # 这里用break的原因是，循环内部还有一个循环，不加是重复的，而且if语句也不会执行
                        break

                # 遍历客体的字符
                for index in range(seq_len - subject_tokens_len + 1):
                    # 判断当前客体截取的字符串和原始客体字符串是否一致
                    if tokens[index:index +
                              subject_tokens_len] == subject_tokens:
                        # 针对之前没有主体的
                        if forbidden_index is None:
                            # 直接就可以进行标签设置了，因为不存在标签重合的情况
                            labels[index][label_subject] = 1
                            # 对接下来的subject_tokens_len进行标签化
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            # 然后终止荀晗
                            break
                        # 针对之前有主体的时候，需要筛选一下
                        # 对于课题在主体之前，或者客体在主体这段字符串之后的才有效
                        elif index < forbidden_index or index >= forbidden_index + len(
                                object_tokens):
                            # 同理打上标签
                            labels[index][label_subject] = 1

                            # 循环接下来的subject_tokens_len歌字符串
                            for i in range(subject_tokens_len - 1):
                                # 对其进行标签化
                                labels[index + i + 1][1] = 1

                            # 终止循环，标签完毕
                            break

    # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
    for i in range(seq_len):
        if labels[i] == [0] * num_labels:
            labels[i][0] = 1

    return labels


def convert_example_to_feature(
        example,
        tokenizer: ErnieTokenizer,
        chineseandpunctuationextractor: ChineseAndPunctuationExtractor,
        label_map,
        max_length: Optional[int] = 512,
        pad_to_max_length: Optional[bool] = None):
    """
    特征转换函数
    :param example: 需要转换的文本
    :param tokenizer: 对应的词表
    :param chineseandpunctuationextractor: 中文标点抽取器
    :param label_map: 标签字典
    :param max_length: 最大长度
    :param pad_to_max_length: 拼接到哦最大长度
    :return: 文本数字特征
    """
    # 获取spo_list对应的value
    """
    {"text": "吴宗宪遭服务生种族歧视, 他气呛: 我买下美国都行!艺人狄莺与孙鹏18岁的独子孙安佐赴美国读高中，没想到短短不到半年竟闹出校园安全事
    件被捕，因为美国正处于校园枪击案频传的敏感时机，加上国外种族歧视严重，外界对于孙安佐的情况感到不乐观 吴宗宪今（30）日录影前谈到美国民情，
    直言国外种族歧视严重，他甚至还被一名墨西哥裔的服务生看不起，让吴宗宪气到喊：「我是吃不起是不是", 
    "spo_list": [{"predicate": "父亲", 
    "object_type": {"@value": "人物"}, "subject_type": "人物", 
    "object": {"@value": "孙鹏"}, "subject": "孙安佐"},
                {"predicate": "母亲", 
    "object_type": {"@value": "人物"}, "subject_type": "人物", 
    "object": {"@value": "狄莺"}, "subject": "孙安佐"}}
    """
    spo_list = example['spo_list'] if "spo_list" in example.keys() else None
    # 获取对应的文本
    text_raw = example['text']

    # 下位词存储器
    sub_text = []
    # 缓存器
    buff = ""
    # 遍历数据每个字符
    for char in text_raw:
        # 诊断该字是不是中文领域的
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            # 判断是空字符吗
            if buff != "":
                # 如果有字符，拼接buff,是一个空格
                sub_text.append(buff)
                # 重置空格
                buff = ""
            # 在拼接字符
            sub_text.append(char)
        else:
            # 针对非中文字符，和之前的字符串项链
            buff += char
    if buff != "":
        # 把非中文的拼接到最后面
        sub_text.append(buff)
    #
    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    # 遍历已经中文字符筛查过的文本
    for (i, token) in enumerate(sub_text):
        # 设置tokens索引
        orig_to_tok_index.append(len(tokens))
        # 获取处理过字符的toekn
        sub_tokens = tokenizer._tokenize(token)
        # 对应词典id拼接
        text_tmp += token
        # 遍历token
        for sub_token in sub_tokens:
            # 针对每一个token开始位置索引
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            # 设置每一个token结束位置索引
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            # 记录对应的token
            tokens.append(sub_token)
            # 针对token大鱼最大限制，则停止
            if len(tokens) >= max_length - 2:
                break
        else:
            # 否则继续
            continue
        break

    # 获取token的长度
    seq_len = len(tokens)
    # 2 tags for each predicate + I tag + O tag
    # 计算对应的标签个数，每个标签起始，还有I和O标签，2N+2
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    # 初始化标签设置为0，每一个字符都是num_labels个标签
    labels = [[0] * num_labels for i in range(seq_len)]
    # 针对spo_list不空的时候
    if spo_list is not None:
        #
        labels = parse_label(spo_list, label_map, tokens, tokenizer)

    # add [CLS] and [SEP] token, they are tagged into "O" for outside
    if seq_len > max_length - 2:
        tokens = tokens[0:(max_length - 2)]
        labels = labels[0:(max_length - 2)]
        tok_to_orig_start_index = tok_to_orig_start_index[0:(max_length - 2)]
        tok_to_orig_end_index = tok_to_orig_end_index[0:(max_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # "O" tag for [PAD], [CLS], [SEP] token
    outside_label = [[1] + [0] * (num_labels - 1)]

    labels = outside_label + labels + outside_label
    tok_to_orig_start_index = [-1] + tok_to_orig_start_index + [-1]
    tok_to_orig_end_index = [-1] + tok_to_orig_end_index + [-1]
    if seq_len < max_length:
        tokens = tokens + ["[PAD]"] * (max_length - seq_len - 2)
        labels = labels + outside_label * (max_length - len(labels))
        tok_to_orig_start_index = tok_to_orig_start_index + [-1] * (
            max_length - len(tok_to_orig_start_index))
        tok_to_orig_end_index = tok_to_orig_end_index + [-1] * (
            max_length - len(tok_to_orig_end_index))

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return InputFeature(
        input_ids=np.array(token_ids),
        seq_len=np.array(seq_len),
        tok_to_orig_start_index=np.array(tok_to_orig_start_index),
        tok_to_orig_end_index=np.array(tok_to_orig_end_index),
        labels=np.array(labels),
    )


class DuIEDataset(paddle.io.Dataset):
    """
    关系抽取数据类
    """

    def __init__(self,
                 data,
                 label_map,
                 tokenizer,
                 max_length=512,
                 pad_to_max_length=False):
        super(DuIEDataset, self).__init__()

        # 数据集
        self.data = data
        # 中文及中文标点符号攀判别器，是compile 函数用于编译正则表达式，生成一个 Pattern 对象
        self.chn_punc_extractor = ChineseAndPunctuationExtractor()
        # 对应的词表
        self.tokenizer = tokenizer
        # 最大的字符长度
        self.max_seq_length = max_length
        # 拼接的最大长度
        self.pad_to_max_length = pad_to_max_length
        # 标签字典
        self.label_map = label_map

    def __len__(self):
        """
        返回数据的大小
        :return:
        """
        return len(self.data)

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        # 加载json数据，并成为一个样本
        example = json.loads(self.data[item])
        # 对数据处理，抓换成特征值
        input_feature = convert_example_to_feature(example, self.tokenizer,
                                                   self.chn_punc_extractor,
                                                   self.label_map,
                                                   self.max_seq_length,
                                                   self.pad_to_max_length)
        return {
            "input_ids":
            np.array(input_feature.input_ids, dtype="int64"),
            "seq_lens":
            np.array(input_feature.seq_len, dtype="int64"),
            "tok_to_orig_start_index":
            np.array(input_feature.tok_to_orig_start_index, dtype="int64"),
            "tok_to_orig_end_index":
            np.array(input_feature.tok_to_orig_end_index, dtype="int64"),
            # If model inputs is generated in `collate_fn`, delete the data type casting.
            "labels":
            np.array(input_feature.labels, dtype="float32"),
        }

    # 类方法可以类名称直接调用，DuIEDataset.from_file()
    @classmethod
    def from_file(cls,
                  file_path: Union[str, os.PathLike],
                  tokenizer: ErnieTokenizer,
                  max_length: Optional[int] = 512,
                  pad_to_max_length: Optional[bool] = None):
        assert os.path.exists(file_path) and os.path.isfile(
            file_path), f"{file_path} dose not exists or is not a file."
        label_map_path = os.path.join(os.path.dirname(file_path),
                                      "predicate2id.json")
        assert os.path.exists(label_map_path) and os.path.isfile(
            label_map_path
        ), f"{label_map_path} dose not exists or is not a file."
        with open(label_map_path, 'r', encoding='utf8') as fp:
            label_map = json.load(fp)
        with open(file_path, "r", encoding="utf-8") as fp:
            data = fp.readlines()
            return cls(data, label_map, tokenizer, max_length,
                       pad_to_max_length)


@dataclass
class DataCollator:
    """
    Collator for DuIE.
    """

    def __call__(self, examples: List[Dict[str, Union[list, np.ndarray]]]):
        batched_input_ids = np.stack([x['input_ids'] for x in examples])
        seq_lens = np.stack([x['seq_lens'] for x in examples])
        tok_to_orig_start_index = np.stack(
            [x['tok_to_orig_start_index'] for x in examples])
        tok_to_orig_end_index = np.stack(
            [x['tok_to_orig_end_index'] for x in examples])
        labels = np.stack([x['labels'] for x in examples])

        return (batched_input_ids, seq_lens, tok_to_orig_start_index,
                tok_to_orig_end_index, labels)


if __name__ == "__main__":
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    d = DuIEDataset.from_file("./data/train_data.json", tokenizer)
    sampler = paddle.io.RandomSampler(data_source=d)
    batch_sampler = paddle.io.BatchSampler(sampler=sampler, batch_size=2)

    collator = DataCollator()
    loader = paddle.io.DataLoader(dataset=d,
                                  batch_sampler=batch_sampler,
                                  collate_fn=collator,
                                  return_list=True)
    for dd in loader():
        model_input = {
            "input_ids": dd[0],
            "seq_len": dd[1],
            "tok_to_orig_start_index": dd[2],
            "tok_to_orig_end_index": dd[3],
            "labels": dd[4]
        }
        print(model_input)
