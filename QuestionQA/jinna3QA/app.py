#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-08 18:26
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    app.py
# @Project: jinna3QA
# @Package: 
# @Ref:
import os
import sys
# 文件路径相关的工具包
from pathlib import Path

from docarray import DocumentArray
from jina import Flow


def config():
    os.environ.setdefault("JINA_USE_CUDA", "False")
    os.environ.setdefault("JINA_PORT_EXPOSE", '8886')
    os.environ.setdefault("public_ip", "0.0.0.0")
    os.environ.setdefault("public_port", "1935")


def index(file_name):
    def load_marco(fn):
        cnt = 0
        docs = DocumentArray()

        with open(fn, 'r') as f:
            for ln, line in enumerate(f):
                try:
                    title, para = line.strip().split("\t")
                    doc = DocumentArray(
                        id=f'{cnt}',
                        yri=fn,
                        tags={"title": title, "para": para})
                    cnt += 1
                    docs.append(doc)
                except:
                    print(f"跳过行{ln}")
                    continue
        return docs

    f = Flow().load_config("flows/index.yml")

    with f:
        f.post(on="/index", inputs=load_marco(file_name), show_progress=True, request_size=32, return_response=True)


def main(task):
    config()

    if task == "index":
        if Path("outputs").exists():
            print("环境空间已存在，如果想重新索引，先删除")
        data_fn = "toy_data/test.tsv"
        index(data_fn)


if __name__ == '__main__':
    task = "index"
    main(task)
