#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-08 14:40
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    query.py
# @Project: faiss_search
# @Package: 
# @Ref:
import json

import requests

TOPK = 5
SERVICE_ADD = 'http://localhost:8888/rocketqa'

while True:
    query = input("请输入一句话:\t")
    if query.strip() == "":
        break

    # 构造数据
    input_data = {}
    input_data['query'] = query
    input_data['topk'] = TOPK
    # 字典转json
    json_str = json.dumps(input_data)

    # 发送post请求
    result = requests.post(SERVICE_ADD, json=input_data)
    # 获取算法检索结果
    res_json = json.loads(result.text)

    print("QUERY:\t" + query)
    for i in range(TOPK):
        title = res_json['answer'][i]["title"]
        para = res_json['answer'][i]["para"]
        score = res_json['answer'][i]["probability"]
        print ('{}'.format(i + 1) + '\t' + title + '\t' + para + '\t' + str(score))
