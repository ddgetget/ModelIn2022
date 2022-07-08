#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-08 14:47
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    rocketqa_service.py
# @Project: faiss_search
# @Package: 
# @Ref:

# 向量检索工具
import faiss


class FaissTool(object):
    def __init__(self, text_filename, index_filename):
        self.engine = faiss.read_index((index_filename))
        self.id2text = []
        for line in open(text_filename):
            # 遍历每一行数据，
            self.id2text.append(line.strip())

    def search(self, q_embs, topk=5):
        # 调检索引擎
        res_dist, res_pid = self.engine.search(q_embs=q_embs, topk=topk)
        result_list = []

        for i in range(topk=topk):
            # faiss返回的事借一个矩阵新的，第一个为0，只返回每条语句的
            result_list.append(self.id2text[res_pid[0][i]])
        return result_list
