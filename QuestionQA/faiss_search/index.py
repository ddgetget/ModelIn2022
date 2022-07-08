#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-08 13:58
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    index.py
# @Project: faiss_search
# @Package: 
# @Ref:
import sys
# 盗取问答工具包
import rocketqa
import faiss
import numpy as np


def build_index(encoder_conf, index_file_name, title_list, para_list):
    # 使用工具加载模型
    dual_encoder = rocketqa.load_model(**encoder_conf)
    # 模型中灌入数据
    para_embs = dual_encoder.encode_para(para=para_list, title=title_list)

    # 记得向量一定要转成ndarray
    para_embs = np.array(list(para_embs))

    print("build index with Faiss...")

    indexer = faiss.IndexFlatIP(768)
    indexer.add(para_embs.astype("float32"))
    faiss.write_index(indexer, index_file_name)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("USAGE: ")
        print("     python3 index.py ${language} ${data_file} ${index_file}")
        print("--For Example:")
        print("     python index.py zh /Users/geng/Documents/data/data10000/dureader.para outputs/test.index")

    language = sys.argv[1]
    data_file = sys.argv[2]
    index_file = sys.argv[3]

    if language == "zh":
        # 针对处理中文问答
        model = "zh_dureader_de_v2"
    elif language == "en":
        model = "v1_marco_de"
    else:
        raise ValueError("目前只支持 (zh)和(en)两种语言")
        exit()

    para_list = []
    title_list = []

    # 按行读取语料
    for line in open(data_file):
        # 对每一行按tab分割成标题和段落
        t, p = line.strip().split("\t")
        para_list.append(p)
        title_list.append(t)

    de_conf = {
        "model": model,
        "use_cuda": False,
        "device_id": 0,
        "batch_size": 32
    }

    build_index(de_conf, index_file, title_list, para_list)
