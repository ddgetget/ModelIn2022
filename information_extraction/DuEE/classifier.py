#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-11 22:37
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    classifier.py
# @Project: DuEE
# @Package: 
# @Ref:

"""
classification
"""
import argparse
import ast
import csv
import json
import os
import random
import traceback
from collections import namedtuple
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer

from utils import read_by_lines, write_by_lines, load_dict

# warnings.filterwarnings('ignore')
"""
For All pre-trained model（English and Chinese),
Please refer to https://paddlenlp.readthedocs.io/zh/latest/model_zoo/index.html#transformer
"""

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=1, help="训词轮训次数")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
parser.add_argument("--tag_path", type=str, default="conf/DuEE-Fin/enum_tag.dict", help="标签路径")
parser.add_argument("--train_data", type=str, default="/Users/geng/data/DuEE-fin/enum/train.tsv", help="训练数据路径")
parser.add_argument("--dev_data", type=str, default="/Users/geng/data/DuEE-fin/enum/dev.tsv", help="验证数据")
parser.add_argument("--test_data", type=str, default="/Users/geng/data/DuEE-fin/enum/test.tsv", help="test data")
parser.add_argument("--predict_data", type=str, default="outputs/test.tsv", help="预测数据")
parser.add_argument("--do_train", type=ast.literal_eval, default=False, help="训练触发器")
parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="预测触发器")
parser.add_argument("--weight_decay", type=float, default=0.01, help="l2正则")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="慢热学习的比例")
parser.add_argument("--max_seq_len", type=int, default=512, help="序列长度")
parser.add_argument("--valid_step", type=int, default=100, help="验证步数")
parser.add_argument("--skip_step", type=int, default=20, help="跳跃步数")
parser.add_argument("--batch_size", type=int, default=2, help="批处理大小")
parser.add_argument("--checkpoints", type=str, default="checkpoints", help="模型保存路径")
parser.add_argument("--init_ckpt", type=str, default="checkpoints/final.pdparams", help="初始化保存模型")
parser.add_argument("--predict_save_path", type=str, default="outputs", help="预测结果保存路径")
parser.add_argument("--seed", type=int, default=1000, help="默认随机种子")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="cpu",
                    help="默认运行运行设备")
args = parser.parse_args()


# yapf: enable.


def set_seed(random_seed):
    """创建随机种子"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    paddle.seed(random_seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accuracy = metric.accumulate()
    metric.reset()
    model.train()
    return float(np.mean(losses)), accuracy


def convert_example(example,
                    tokenizer,
                    label_map=None,
                    max_seq_len=512,
                    is_test=False):
    """convert_example"""
    has_text_b = False
    if isinstance(example, dict):
        has_text_b = "text_b" in example.keys()
    else:
        has_text_b = "text_b" in example._fields

    text_b = None
    if has_text_b:
        text_b = example.text_b

    tokenized_input = tokenizer(text=example.text_a,
                                text_pair=text_b,
                                max_seq_len=max_seq_len)
    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']

    if is_test:
        return input_ids, token_type_ids
    else:
        label = np.array([label_map[example.label]], dtype="int64")
        return input_ids, token_type_ids, label


class DuEventExtraction(paddle.io.Dataset):
    """Du"""

    def __init__(self, data_path, tag_path):
        self.label_vocab = load_dict(tag_path)
        self.examples = self._read_tsv(data_path)

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            Example = namedtuple('Example', headers)
            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        line[index] = text
                try:
                    example = Example(*line)
                except Exception as e:
                    traceback.print_exc()
                    raise Exception(e)
                examples.append(example)
            return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def data_2_examples(datas):
    """data_2_examples"""
    # 分配两个空间
    has_text_b, examples = False, []
    # 判断数据是列表类型
    if isinstance(datas[0], list):
        # TODO 需要查询这个namedtuple是什么意思
        Example = namedtuple('Example', ["text_a", "text_b"])
        has_text_b = True
    else:
        Example = namedtuple('Example', ["text_a"])
    for item in datas:
        if has_text_b:
            example = Example(text_a=item[0], text_b=item[1])
        else:
            example = Example(text_a=item)
        examples.append(example)
    return examples


def do_train():
    # 设置程序运行平台
    paddle.set_device(args.device)
    # 获取单词大小
    world_size = paddle.distributed.get_world_size()
    rank = paddle.distributed.get_rank()
    if world_size > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)
    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}

    # 加载预训练模型
    model = ErnieForSequenceClassification.from_pretrained(
        "ernie-1.0", num_classes=len(label_map))
    # 将模型转成layer
    model = paddle.DataParallel(model)
    # 获取模型对应的词表
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")

    print("============start train==========")
    # 获取训练数据
    train_ds = DuEventExtraction(args.train_data, args.tag_path)
    # 获取验证数据
    dev_ds = DuEventExtraction(args.dev_data, args.tag_path)
    # 获取测试数据
    test_ds = DuEventExtraction(args.test_data, args.tag_path)

    # 柯里化，将数据转成向量
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         label_map=label_map,
                         max_seq_len=args.max_seq_len)

    # 数据在函数中转换
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),
        Stack(dtype="int64")  # label
    ): fn(list(map(trans_func, samples)))

    #分布式批采样器加载数据的一个子集。每个进程可以传递给DataLoaderDistributedBatchSampler的实例，每个进程加载原始数据的一个子集。
    batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    train_loader = paddle.io.DataLoader(dataset=train_ds,
                                        batch_sampler=batch_sampler,
                                        collate_fn=batchify_fn)
    dev_loader = paddle.io.DataLoader(dataset=dev_ds,
                                      batch_size=args.batch_size,
                                      collate_fn=batchify_fn)
    test_loader = paddle.io.DataLoader(dataset=test_ds,
                                       batch_size=args.batch_size,
                                       collate_fn=batchify_fn)

    # 计算总的运行步数
    num_training_steps = len(train_loader) * args.num_epoch
    # 获取准确率计算函数
    metric = paddle.metric.Accuracy()
    criterion = paddle.nn.loss.CrossEntropyLoss()
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    step, best_performerence = 0, 0.0
    model.train()
    for epoch in range(args.num_epoch):
        for idx, (input_ids, token_type_ids, labels) in enumerate(train_loader):
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss_item = loss.numpy().item()
            if step > 0 and step % args.skip_step == 0 and rank == 0:
                print(f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) ' \
                      f'- loss: {loss_item:.6f} acc {acc:.5f}')
            if step > 0 and step % args.valid_step == 0 and rank == 0:
                loss_dev, acc_dev = evaluate(model, criterion, metric,
                                             dev_loader)
                print(f'dev step: {step} - loss: {loss_dev:.6f} accuracy: {acc_dev:.5f}, ' \
                      f'current best {best_performerence:.5f}')
                if acc_dev > best_performerence:
                    best_performerence = acc_dev
                    print(f'==============================================save best model ' \
                          f'best performerence {best_performerence:5f}')
                    paddle.save(model.state_dict(),
                                '{}/best.pdparams'.format(args.checkpoints))
            step += 1

    # save the final model
    if rank == 0:
        paddle.save(model.state_dict(),
                    '{}/final.pdparams'.format(args.checkpoints))


def do_predict():
    set_seed(args.seed)
    paddle.set_device(args.device)

    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}

    model = ErnieForSequenceClassification.from_pretrained(
        "ernie-1.0", num_classes=len(label_map))
    model = paddle.DataParallel(model)
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")

    print("============start predict==========")
    if not args.init_ckpt or not os.path.isfile(args.init_ckpt):
        raise Exception("init checkpoints {} not exist".format(args.init_ckpt))
    else:
        state_dict = paddle.load(args.init_ckpt)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.init_ckpt)

    # load data from predict file
    sentences = read_by_lines(args.predict_data)  # origin data format
    sentences = [json.loads(sent) for sent in sentences]

    encoded_inputs_list = []
    for sent in sentences:
        sent = sent["text"]
        input_sent = [sent]  # only text_a
        if "text_b" in sent:
            input_sent = [[sent, sent["text_b"]]]  # add text_b
        example = data_2_examples(input_sent)[0]
        input_ids, token_type_ids = convert_example(
            example, tokenizer, max_seq_len=args.max_seq_len, is_test=True)
        encoded_inputs_list.append((input_ids, token_type_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),
    ): fn(samples)
    # Seperates data into some batches.
    batch_encoded_inputs = [
        encoded_inputs_list[i:i + args.batch_size]
        for i in range(0, len(encoded_inputs_list), args.batch_size)
    ]
    results = []
    model.eval()
    for batch in batch_encoded_inputs:
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        probs_ids = paddle.argmax(probs, -1).numpy()
        probs = probs.numpy()
        for prob_one, p_id in zip(probs.tolist(), probs_ids.tolist()):
            label_probs = {}
            for idx, p in enumerate(prob_one):
                label_probs[id2label[idx]] = p
            results.append({"probs": label_probs, "label": id2label[p_id]})

    assert len(results) == len(sentences)
    for sent, ret in zip(sentences, results):
        sent["pred"] = ret
    sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
    write_by_lines(args.predict_save_path, sentences)
    print("save data {} to {}".format(len(sentences), args.predict_save_path))


if __name__ == '__main__':

    if args.do_train:
        do_train()
    elif args.do_predict:
        do_predict()
