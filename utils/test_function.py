#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-03-03 11:57
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    test_function.py
# @Project: ModelIn2022
# @Package:


def test_if(i=0):
    i = 89
    print(i if i == 1 else 0)


def test_dict(param):
    print(param["name"] if "name" in param.keys() else "default")
    for key, value in param.items():
        print(key, value)


if __name__ == '__main__':
    # test_if(78)
    test_dict({"id": 12, "age": 89})
