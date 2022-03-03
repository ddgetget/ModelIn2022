#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-03-03 12:34
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    MainApplication.py
# @Project: ModelIn2022
# @Package:
from ComputerVision.ImageClassificationFromScratch import CVModel

if __name__ == '__main__':
    info = {"version": "V-2.0","function":"This module for cv "}

    params = {}
    model = CVModel(params)
    desc = model.describe(info=info, export=True)
    print(desc)
