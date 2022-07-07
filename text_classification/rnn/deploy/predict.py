# -*- encoding: utf-8 -*-
'''
@File    :   predict.py
@Time    :   2022/07/07 19:22:33
@Author  :   LongGengYung
@Version :   1.0
@Contact :   yonglonggeng@163.com
@License :   (C)Copyright 1993-2022, Liugroup-NLPR-CASIA
@Desc    :   None
@WeChat  :   superior_god
@微信公众号:   庚庚体验馆
@知乎     :   雍珑庚
'''

# here put the import lib

import argparse

import paddle
from paddlenlp.data import Tuple, Pad, Stack, Vocab, JiebaTokenizer
from scipy.special import softmax

# 科学计算包
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_file", type=str,default='../outputs/static_graph_params.pdmodel', help="静态模型保存路径")
parser.add_argument("--params_file", type=str,default='../outputs/static_graph_params.pdiparams', help="静态图保存路径")

parser.add_argument('--network', choices=['bow', 'lstm', 'bilstm', 'gru', 'bigru','rnn', 'birnn', 'bilstm_attn', 'cnn', 'textcnn'], default="bow",
                    help="选择使用的网络")
parser.add_argument("--vocab_path", type=str, default="../outputs/vocab.json", help="词典库")
parser.add_argument("--max_seq_length",default=128, type=int, help="最大序列长度")
parser.add_argument("--batch_size", default=2, type=int, help="批次")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'],default="cpu", help="选择所使用的设备")
args = parser.parse_args()


def preprocess_prediction_data(text, tokenizer):
    # 获取文字对应的id
    input_id = tokenizer.encode(text)
    # 获取序列的长度
    seq_len = len(input_id)

    return input_id, seq_len


class Predictor(object):
    def __init__(self, model_file, params_file, device, max_seq_length):
        self.max_seq_length = max_seq_length
        # 获取推理脚本的配置文件
        config = paddle.inference.Config(model_file, params_file)
        if device == "cpu":
            config.disable_gpu()
        elif device == "gpu":
            config.enable_use_gpu()
        elif device == "xpu":
            config.enable_xpu()

        # 设置配置文件
        config.switch_use_feed_fetch_ops(False)

        self.predictor = paddle.inference.create_predictor(config)

        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]

        self.output_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0]
        )

    def predict(self, data, tokenizer, label_map, batch_size=1, network="bow"):
        """
        对数据预测
        :param data:
        :param tokenizer:
        :param label_map:
        :param batch_size:
        :param network:
        :return:
        """

        examples = []
        for text in data:
            # 对每一个query今夕进行转换
            input_id, seq_len = preprocess_prediction_data(text, tokenizer)
            examples.append((input_id, seq_len))

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=tokenizer.vocab.token_to_idx.get("[PAD]", 0)),  # input_id
            Stack()  # seq_leb
        ): fn(samples)

        # 分割数据成多个批次、
        batches = [
            examples[idx:idx + batch_size] for idx in range(0, len(examples), batch_size)
        ]

        results = []
        for batch in batches:
            # 获取一个batch的数据
            input_ids, seq_lens = batchify_fn(batch)
            # TODO 这一句暂时还不清楚
            self.input_handles[0].copy_from_cpu(input_ids)

            if network in [
                "lstm", "bilstm", "gru", "bigru", "rnn", "birnn",
                "bilstm_attn"
            ]: self.input_handles[1].copy_from_cpu(seq_lens)  # TODO 这一句也不清楚干嘛

            self.predictor.run()
            # 计算
            logits = self.output_handle.copy_to_cpu()
            probs = softmax(logits, axis=1)
            idx = np.argmax(probs, axis=1)
            idx = idx.tolist()

            # 计算映射的标签
            labels = [label_map[i] for i in idx]
            # 一个批次的拼接起来
            results.extend(labels)

        return results


if __name__ == '__main__':
    # 定义预测器
    predictor = Predictor(args.model_file, args.params_file, args.device, args.max_seq_length)

    # 定义部分数据
    data = [
        '非常不错，服务很好，位于市中心区，交通方便，不过价格也高！',
        '怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片',
        '作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。',
    ]

    # 导入词典
    vocab = Vocab.from_json(args.vocab_path)
    # 将辞典专案成数字
    tokenizer = JiebaTokenizer(vocab)
    # 定义标签对应表
    label_map = {0: 'negative', 1: 'positive'}
    # 获取预测结果
    results = predictor.predict(data=data, tokenizer=tokenizer, label_map=label_map, batch_size=args.batch_size,
                                network=args.network)

    # 按行便利，打印预测结果
    for idx, text in enumerate(data):
        print('Data: {} \t Label: {}'.format(text, results[idx]))
