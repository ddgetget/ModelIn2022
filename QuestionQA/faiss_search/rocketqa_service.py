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
import json

import faiss

# 导入问答工具
import rocketqa

# 导入后端服务器工具
from tornado import web, ioloop

# 科学计算包
import numpy as np


class FaissTool(object):
    def __init__(self, text_filename, index_filename):
        self.engine = faiss.read_index((index_filename))
        self.id2text = []
        for line in open(text_filename):
            # 遍历每一行数据，
            self.id2text.append(line.strip())

    def search(self, q_embs, topk=5):
        # 调检索引擎
        res_dist, res_pid = self.engine.search(q_embs, topk)
        print(res_pid)
        print(res_dist)
        result_list = []

        for i in range(topk):
            # faiss返回的事借一个矩阵新的，第一个为0，只返回每条语句的
            result_list.append(self.id2text[res_pid[0][i]])
        return result_list


def create_rocket_app(sub_address, rocketqa_server, lanaguae, data_flile, index_file):
    if lanaguae == "zh":
        de_model = 'zh_dureader_de_v2'
        ce_model = 'zh_dureader_ce_v2'
    elif lanaguae == "en":
        de_model = 'v1_marco_de'
        ce_model = 'v1_marco_ce'
    else:
        print("其他模型暂时不支持")
        raise ValueError("%s模型暂时不支持" % lanaguae)

    de_conf = {
        "model": de_model,
        "use_cuda": False,
        "device_id": 0,
        "batch_size": 32
    }
    ce_conf = {
        "model": ce_model,
        "use_cuda": False,
        "device_id": 0,
        "batch_size": 32
    }

    # 加载模型

    dual_encoder = rocketqa.load_model(**de_conf)
    cross_encoder = rocketqa.load_model(**ce_conf)

    # 构建faisssuo搜索工具
    print("faiss索引已构建")
    faiss_tool = FaissTool(data_flile, index_file)

    return web.Application([(sub_address, rocketqa_server,
                             dict(faiss_tools=faiss_tool, dual_encoder=dual_encoder, cross_encoder=cross_encoder))])


class RocketQAServer(web.RequestHandler):
    def __init__(self, application, request, **kwargs):
        web.RequestHandler.__init__(self, application, request)
        self._faiss_tool = kwargs['faiss_tools']
        self._dual_encoder = kwargs['dual_encoder']
        self._cross_encoder = kwargs['cross_encoder']

    def get(self):
        """

        :return:
        """
        pass

    def post(self):
        # 获取请求体内容
        input_request = self.request.body

        # 构建返回体数据格式
        output = {}
        output['error_code'] = 0
        output['error_message'] = ''
        output['answer'] = []

        if input_request is None:
            # 针对请求体为空的类型
            output['error_code'] = 1
            output['error_message'] = "输入为空"
            self.write(json.dumps(output))

        # 判断获取数据是否为有效数据
        try:
            # 加载json数据
            input_data = json.loads(input_request)
        except:
            output['error_message'] = "数据不完整"
            output['error_code'] = 2
            self.write(json.dumps(output))

        # 判断query字段是否在请求体内
        if "query" not in input_data:
            output['error_message'] = "query语句丢失"
            output['error_code'] = 3

            self.write(json.dumps(output))
            return

        query = input_data['query']
        # 设置默认topk为5
        topk = 5
        if "topk" in input_data:
            # 如果请求体有topk，则拿新的
            topk = input_data["topk"]

        # 编码query
        q_embs = self._dual_encoder.encode_query(query=[query])
        # 获取的向量需要转成ndarray类型
        q_embs = np.array(list(q_embs))

        # 是用faiss进行搜索
        search_result = self._faiss_tool.search(q_embs, topk)

        # 组织结果格式
        titles = []
        paras = []
        queries = []

        for t_p in search_result:
            queries.append(query)
            # 对标题和段落进行分裂
            t, p = t_p.split("\t")
            titles.append(t)
            paras.append(p)

        # 对结果与原句子进行打分
        ranking_score = self._cross_encoder.matching(query=queries, para=paras, title=titles)
        # 类型转换
        ranking_score = list(ranking_score)

        final_result = {}
        for i in range(len(paras)):
            # 输入的query和计算出来的进行匹配分数赋值
            final_result[query + "\t" + titles[i] + "\t" + paras[i]] = ranking_score[i]

        # 对每一个价值对按照value降序排列
        sort_res = sorted(final_result.items(), key=lambda a: a[1], reverse=True)

        # 对结果集进一步处理
        for qtp, score in sort_res:
            # 便利key和value
            one_answer = {}
            one_answer['probability'] = score
            q, t, p = qtp.split("\t")
            one_answer['title'] = t
            one_answer['para'] = p
            # 将一条结果拼接到结果上
            output['answer'].append(one_answer)

        result_str = json.dumps(output, ensure_ascii=False)
        self.write(result_str)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="针对的语种")
    parser.add_argument("--data_file", type=str, default="/Users/geng/Documents/data/data10000/dureader.para",
                        help="训练的数据路径")
    parser.add_argument("--index_file", type=str, default="outputs/test.index", help="向量索引路径")

    args = parser.parse_args()

    if args.language != "en" and args.language != "zh":
        print("不合法的输入")
        exit()

    # 定义路由url
    sub_address = r"/rocketqa"
    # 定义端口号
    port = '8888'
    # 创建服务
    app = create_rocket_app(sub_address, RocketQAServer, args.language, args.data_file, args.index_file)

    # 监听端口号
    app.listen(port)
    # 启动服务
    ioloop.IOLoop.current().start()
