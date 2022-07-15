#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-12 21:41
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    extract_chinese_and_punct.py
# @Project: DuIE
# @Package: 
# @Ref:

import re

LHan = [
    [0x2E80, 0x2E99],  # Han # So  [26] CJK RADICAL REPEAT, CJK RADICAL RAP
    [0x2E9B, 0x2EF3
     ],  # Han # So  [89] CJK RADICAL CHOKE, CJK RADICAL C-SIMPLIFIED TURTLE
    [0x2F00, 0x2FD5],  # Han # So [214] KANGXI RADICAL ONE, KANGXI RADICAL FLUTE
    0x3005,  # Han # Lm       IDEOGRAPHIC ITERATION MARK
    0x3007,  # Han # Nl       IDEOGRAPHIC NUMBER ZERO
    [0x3021,
     0x3029],  # Han # Nl   [9] HANGZHOU NUMERAL ONE, HANGZHOU NUMERAL NINE
    [0x3038,
     0x303A],  # Han # Nl   [3] HANGZHOU NUMERAL TEN, HANGZHOU NUMERAL THIRTY
    0x303B,  # Han # Lm       VERTICAL IDEOGRAPHIC ITERATION MARK
    [
        0x3400, 0x4DB5
    ],  # Han # Lo [6582] CJK UNIFIED IDEOGRAPH-3400, CJK UNIFIED IDEOGRAPH-4DB5
    [
        0x4E00, 0x9FC3
    ],  # Han # Lo [20932] CJK UNIFIED IDEOGRAPH-4E00, CJK UNIFIED IDEOGRAPH-9FC3
    [
        0xF900, 0xFA2D
    ],  # Han # Lo [302] CJK COMPATIBILITY IDEOGRAPH-F900, CJK COMPATIBILITY IDEOGRAPH-FA2D
    [
        0xFA30, 0xFA6A
    ],  # Han # Lo  [59] CJK COMPATIBILITY IDEOGRAPH-FA30, CJK COMPATIBILITY IDEOGRAPH-FA6A
    [
        0xFA70, 0xFAD9
    ],  # Han # Lo [106] CJK COMPATIBILITY IDEOGRAPH-FA70, CJK COMPATIBILITY IDEOGRAPH-FAD9
    [
        0x20000, 0x2A6D6
    ],  # Han # Lo [42711] CJK UNIFIED IDEOGRAPH-20000, CJK UNIFIED IDEOGRAPH-2A6D6
    [0x2F800, 0x2FA1D]
]  # Han # Lo [542] CJK COMPATIBILITY IDEOGRAPH-2F800, CJK COMPATIBILITY IDEOGRAPH-2FA1D

# 中文标点符号
CN_PUNCTS = [(0x3002, "。"), (0xFF1F, "？"), (0xFF01, "！"), (0xFF0C, "，"),
             (0x3001, "、"), (0xFF1B, "；"), (0xFF1A, "："), (0x300C, "「"),
             (0x300D, "」"), (0x300E, "『"), (0x300F, "』"), (0x2018, "‘"),
             (0x2019, "’"), (0x201C, "“"), (0x201D, "”"), (0xFF08, "（"),
             (0xFF09, "）"), (0x3014, "〔"), (0x3015, "〕"), (0x3010, "【"),
             (0x3011, "】"), (0x2014, "—"), (0x2026, "…"), (0x2013, "–"),
             (0xFF0E, "．"), (0x300A, "《"), (0x300B, "》"), (0x3008, "〈"),
             (0x3009, "〉"), (0x2015, "―"), (0xff0d, "－"), (0x0020, " ")]
#(0xFF5E, "～"),
# 英文标点符号
EN_PUNCTS = [[0x0021, 0x002F], [0x003A, 0x0040], [0x005B, 0x0060],
             [0x007B, 0x007E]]


class ChineseAndPunctuationExtractor(object):
    """
    中文和标点符号的抽取器
    """

    def __init__(self):

        # 构建匹配规则
        self.chinese_re = self.build_re()

    def is_chinese_or_punct(self, c):


        # 判断是否匹配
        if self.chinese_re.match(c):
            # 如果匹配，返回True
            return True
        else:
            return False

    def build_re(self):
        # 声明一个列表存储器
        L = []

        # 遍历特殊字符
        for i in LHan:
            # 判断字符是否为列表类型
            if isinstance(i, list):
                # 真毒列表类型，字符拆解
                f, t = i
                try:
                    # 获取f对应的ASCII
                    f = chr(f)
                    # 获取t对应的ASCII
                    t = chr(t)
                    # 将两个ASCII用-拼接
                    L.append('%s-%s' % (f, t))
                except:
                    pass  # A narrow python build, so can't use chars > 65535 without surrogate pairs!

            else:
                # 针对不是列表型的
                try:
                    # 试着拼接这个字符的ASCII
                    L.append(chr(i))
                except:
                    pass

        # 遍历中文符号
        for j, _ in CN_PUNCTS:
            try:
                # 获取点符号的ASCII
                L.append(chr(j))
            except:
                pass

        # 这对英文标点符号
        for k in EN_PUNCTS:
            # 获取f,t
            f, t = k
            try:
                # 获取对应的ASCII
                f = chr(f)
                t = chr(t)
                # 拼接结果
                L.append('%s-%s' % (f, t))
            except:
                raise ValueError()
                pass  # A narrow python build, so can't use chars > 65535 without surrogate pairs!

        RE = '[%s]' % ''.join(L)
        # print('RE:', RE.encode('utf-8'))
        # 正则匹配，compile 函数用于编译正则表达式，生成一个 Pattern 对象
        return re.compile(RE, re.UNICODE)


if __name__ == '__main__':
    # 声明一个中文和标点符号抽取器
    extractor = ChineseAndPunctuationExtractor()
    # 遍历当前query
    for c in "韩邦庆（1856～1894）曾用名寄，字子云，别署太仙、大一山人、花也怜侬、三庆":
        # 判断当前字符是否在抽取器里面
        if extractor.is_chinese_or_punct(c):
            # 如果是，输出yes
            print(c, 'yes')
        else:
            # 否则输出no
            print(c, "no")

    print("～", extractor.is_chinese_or_punct("～"))
    print("~", extractor.is_chinese_or_punct("~"))
    print("―", extractor.is_chinese_or_punct("―"))
    print("-", extractor.is_chinese_or_punct("-"))
