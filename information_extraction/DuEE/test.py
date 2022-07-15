#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-12 17:49
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    test.py
# @Project: DuEE
# @Package: 
# @Ref:


def tets_list():
    ns = ["asd", 'bias', 'norm', 'asdsadas', 'asd']
    for n in ns:
        for nd in ["bias", "norm"]:
            if not any(nd in n):
                print(n)


if __name__ == '__main__':
    tets_list()
