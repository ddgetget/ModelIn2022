#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-11 22:34
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    sequence_labeling.py
# @Project: DuEE
# @Package: 
# @Ref:


"""
sequence labeling
"""
import argparse
import ast
import json
import os
import random
import warnings
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification

from utils import read_by_lines, write_by_lines, load_dict

warnings.filterwarnings('ignore')

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=1, help="训练次数")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
parser.add_argument("--tag_path", type=str, default="conf/DuEE-Fin/trigger_tag.dict", help="标签文件路径")
parser.add_argument("--train_data", type=str, default="/Users/geng/Documents/data/DuEE-fin/trigger/train.tsv",
                    help="训练数据")
parser.add_argument("--dev_data", type=str, default="/Users/geng/Documents/data/DuEE-fin/trigger/dev.tsv", help="验证数据")
parser.add_argument("--test_data", type=str, default="/Users/geng/Documents/data/DuEE-fin/trigger/test.tsv",
                    help="测试数据")
parser.add_argument("--predict_data", type=str, default=None, help="预测数据")
parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="是否训练")
parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="是否预测")
parser.add_argument("--weight_decay", type=float, default=0.01,
                    help="权重衰减系数，是一个float类型或者shape为[1] ，数据类型为float32的Tensor类型。默认值为0.01。")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="待定")
parser.add_argument("--max_seq_len", type=int, default=512, help="能处理的序列最大长度")
parser.add_argument("--valid_step", type=int, default=100, help="验证步数")
parser.add_argument("--skip_step", type=int, default=20, help="跳跃步数")
parser.add_argument("--batch_size", type=int, default=2, help="训练每个批次打大小")
parser.add_argument("--checkpoints", type=str, default="checkpoints/DuEE-Fin", help="模型存取")
parser.add_argument("--init_ckpt", type=str, default="outputs", help="已经训练好的模型")
parser.add_argument("--predict_save_path", type=str, default="outputs", help="预测结果保存路径")
parser.add_argument("--seed", type=int, default=1000, help="随机运行的种子")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="cpu",
                    help="选择运行的设备，默认在CPU上运行")
args = parser.parse_args()


# yapf: enable.


def set_seed(args):
    """sets random seed"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, num_label, data_loader):
    """evaluate"""
    model.eval()
    metric.reset()
    losses = []
    for input_ids, seg_ids, seq_lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        loss = paddle.mean(
            criterion(logits.reshape([-1, num_label]), labels.reshape([-1])))
        losses.append(loss.numpy())
        preds = paddle.argmax(logits, axis=-1)
        n_infer, n_label, n_correct = metric.compute(None, seq_lens, preds,
                                                     labels)
        metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        precision, recall, f1_score = metric.accumulate()
    avg_loss = np.mean(losses)
    model.train()

    return precision, recall, f1_score, avg_loss


def convert_example_to_feature(example,
                               tokenizer,
                               label_vocab=None,
                               max_seq_len=512,
                               no_entity_label="O",
                               ignore_label=-1,
                               is_test=False):
    tokens, labels = example
    tokenized_input = tokenizer(tokens,
                                return_length=True,
                                is_split_into_words=True,
                                max_seq_len=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = tokenized_input['seq_len']

    if is_test:
        return input_ids, token_type_ids, seq_len
    elif label_vocab is not None:
        labels = labels[:(max_seq_len - 2)]
        encoded_label = [no_entity_label] + labels + [no_entity_label]
        encoded_label = [label_vocab[x] for x in encoded_label]
        return input_ids, token_type_ids, seq_len, encoded_label


class DuEventExtraction(paddle.io.Dataset):
    """DuEventExtraction"""

    def __init__(self, data_path, tag_path):
        self.label_vocab = load_dict(tag_path)
        self.word_ids = []
        self.label_ids = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            # skip the head line
            next(fp)
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                self.word_ids.append(words)
                self.label_ids.append(labels)
        self.label_num = max(self.label_vocab.values()) + 1

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, index):
        return self.word_ids[index], self.label_ids[index]


def do_train():
    # 设置项目运行的平台
    paddle.set_device(args.device)
    # TODO 需要查询paddle.distributed.get_world_size这个API的用法
    world_size = paddle.distributed.get_world_size()
    # TODO 需要查询paddle.distributed.get_rank的用法
    rank = paddle.distributed.get_rank()
    # 如果有单词，那么进行环境初始化
    if world_size > 1:
        paddle.distributed.init_parallel_env()

    # 设置所有用到包的随机种子
    set_seed(args)

    # 对于没有标签的用O来表示
    no_entity_label = "O"
    # 需要忽略的标签用-1表示
    ignore_label = -1

    # 获取ernie的词表解析器
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    # 获取当前所有的标签
    label_map = load_dict(args.tag_path)
    # 标签和ID翻转
    id2label = {val: key for key, val in label_map.items()}
    # 加载分类预训练模型
    # 这个模型构面添加了dropout,linear层，以达到目标类别个数
    model = ErnieForTokenClassification.from_pretrained(
        "ernie-1.0", num_classes=len(label_map))
    # 详细描述：https://ew6tx9bj6e.feishu.cn/docx/doxcnB0u7eOZ8Q6WUTa2rgym94W
    model = paddle.DataParallel(model)

    print("============start train==========")
    # 加载训练数据
    train_ds = DuEventExtraction(args.train_data, args.tag_path)
    # 加载验证数据
    dev_ds = DuEventExtraction(args.dev_data, args.tag_path)
    # 加载测试数据
    test_ds = DuEventExtraction(args.test_data, args.tag_path)


    # TODO 这个函数还需要有待详细查询
    """
    偏函数的作用：和装饰器一样，它可以扩展函数的功能，但又不完成等价于装饰器。通常应用的场景是当我们要频繁调用某个函数时，其中某些参数是已知
    的固定值，通常我们可以调用这个函数多次，但这样看上去似乎代码有些冗余，而偏函数的出现就是为了很少的解决这一个问题。举一个很简单的例子，比
    如我就想知道 100 加任意数的和是多少，通常我们的实现方式是这样的：
    
    # 第一种做法：
    def add(*args):
        return sum(args)
    
    print(add(1, 2, 3) + 100)
    print(add(5, 5, 5) + 100)
    
    # 第二种做法
    def add(*args):
        # 对传入的数值相加后，再加上100返回
        return sum(args) + 100
    
    print(add(1, 2, 3))  # 106
    print(add(5, 5, 5))  # 115 
    
    看上面的代码，貌似也挺简单，也不是很费劲。但两种做法都会存在有问题：第一种，100这个固定值会返回出现，代码总感觉有重复；
    第二种，就是当我们想要修改 100 这个固定值的时候，我们需要改动 add 这个方法。下面我们来看下用 parital 怎么实现：

    from functools import partial
    
    def add(*args):
        return sum(args)
    
    add_100 = partial(add, 100)
    print(add_100(1, 2, 3))  # 106
    
    add_101 = partial(add, 101)
    print(add_101(1, 2, 3))  # 107
    """
    # 柯里化将数据转换成特征向量
    trans_func = partial(convert_example_to_feature,
                         tokenizer=tokenizer,
                         label_vocab=train_ds.label_vocab,
                         max_seq_len=args.max_seq_len,
                         no_entity_label=no_entity_label,
                         ignore_label=ignore_label,
                         is_test=False)
    # lambd函数
    """
    将lambda函数赋值给一个变量，通过这个变量间接调用该lambda函数。
    def sum(x,y):
        return x+y
    print(sum(1,2))
    
    使用lambda函数
    sum = lambda x,y : x+y
    print(sum(1,2))
    """

    """
    map函数：map(function, iterable, ...)
    >>> def square(x) :         # 计算平方数
    ...     return x ** 2
    ... 
    >>> map(square, [1,2,3,4,5])    # 计算列表各个元素的平方
    <map object at 0x100d3d550>     # 返回迭代器
    >>> list(map(square, [1,2,3,4,5]))   # 使用 list() 转换为列表
    [1, 4, 9, 16, 25]
    >>> list(map(lambda x: x ** 2, [1, 2, 3, 4, 5]))   # 使用 lambda 匿名函数
    [1, 4, 9, 16, 25]
    >>> 
    
    
    """
    batchify_fn = lambda samples, fn=Tuple(  # 这里是一个元组，
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # 输入文字的id
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # 输入文字类型的id
        Stack(dtype='int64'),  # 序列的长度
        Pad(axis=0, pad_val=ignore_label, dtype='int64')  # 标签
    ): fn(list(map(trans_func, samples)))  # 数据转张量函数 和 样本数据
    # 【注意📢】冒号前都是参数哈，冒号后是函数逻辑

    # 返回样本下标数组的迭代器
    batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)

    # DataLoader，迭代 dataset 数据的迭代器，迭代器返回的数据中的每个元素都是一个Tensor。
    # 这里batch_sampler已经是数字了，
    train_loader = paddle.io.DataLoader(dataset=train_ds,
                                        batch_sampler=batch_sampler,
                                        collate_fn=batchify_fn)

    dev_loader = paddle.io.DataLoader(dataset=dev_ds,
                                      batch_size=args.batch_size,
                                      collate_fn=batchify_fn)

    # 目测这里的测试数据集没有用到
    test_loader = paddle.io.DataLoader(dataset=test_ds,
                                       batch_size=args.batch_size,
                                       collate_fn=batchify_fn)

    # 计算训练的步数
    # 公式：训练步数=训练接长度*迭代次数
    num_training_steps = len(train_loader) * args.num_epoch

    # 生成参数，遍历模型测参数名称，排除掉带偏置，正则的参数
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    # 优化器部分，使用AdamW
    # AdamW优化器出自 DECOUPLED WEIGHT DECAY REGULARIZATION，用来解决 Adam 优化器中L2正则化失效的问题。
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,  # 权重衰减系数，是一个float类型或者shape为[1] ，数据类型为float32的Tensor类型。默认值为0.01。
        apply_decay_param_fun=lambda x: x in decay_params)  # 传入函数时，只有可以使 apply_decay_param_fun(Tensor.name)==True
    # 的Tensor会进行weight decay更新。只有在想要指定特定需要进行
    # weight decay更新的参数时使用。默认值为None。
    """
    from paddlenlp.metrics import ChunkEvaluator

    num_infer_chunks = 10
    num_label_chunks = 9
    num_correct_chunks = 8

    label_list = [1,1,0,0,1,0,1]
    evaluator = ChunkEvaluator(label_list)
    evaluator.update(num_infer_chunks, num_label_chunks, num_correct_chunks)
    precision, recall, f1 = evaluator.accumulate()
    print(precision, recall, f1)
    # 0.8 0.8888888888888888 0.8421052631578948
    """
    # 这里读取是information_extraction/DuEE/conf/DuEE-Fin/trigger_tag.dict这个字典，keys是数字
    metric = ChunkEvaluator(label_list=train_ds.label_vocab.keys(),
                            suffix=False)

    # 定义损失函数
    # 该OP计算输入input和标签label间的交叉熵损失 ，它结合了 LogSoftmax 和 NLLLoss 的OP计算，可用于训练一个 n 类分类器。
    # from paddle.nn import CrossEntropyLoss
    criterion = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    # 初始化迭代步数和f1-score
    step, best_f1 = 0, 0.0
    # 模型训练，这里是之前的那个layer层
    model.train()

    # 根据迭代次数循环
    for epoch in range(args.num_epoch):
        # 以dataloader一个进行遍历
        # 包括：输入文字的id,输入文字的类型id,序列的长度，以及标签
        tag=0 # tag测试用
        for idx, (input_ids, token_type_ids, seq_lens,
                  labels) in enumerate(train_loader):

            tag+=1
            if tag%50==0:print("第",epoch,"遍","标志位：",tag)
            if tag>200:break

            # 根据模型计算输出层的结果，是一个有类别个数个linear,需要把最后一层转换成标签个数个，其实这块本来输出也是标签个，但是是有批次的
            logits = model(input_ids,
                           token_type_ids).reshape([-1, train_ds.label_num])
            # 使用交叉熵计算标签的值和真实值之间的差距
            loss = paddle.mean(criterion(logits, labels.reshape([-1])))
            # 利用框架反向求导
            loss.backward()
            # 使用优化器
            optimizer.step()
            # 清除之前的梯度
            optimizer.clear_grad()

            # item返回指定索引处的值
            loss_item = loss.numpy().item()
            # 跳跃式打印日志，打印训练的
            if step > 0 and step % args.skip_step == 0 and rank == 0:
                print(
                    f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) - loss: {loss_item:.6f}'
                )

            # 跳跃式打印验证部分
            if step > 0 and step % args.valid_step == 0 and rank == 0:
                # 到特定步数，开始调用验证数据
                p, r, f1, avg_loss = evaluate(model, criterion, metric,
                                              len(label_map), dev_loader)
                print(f'dev step: {step} - loss: {avg_loss:.5f}, precision: {p:.5f}, recall: {r:.5f}, ' \
                      f'f1: {f1:.5f} current best {best_f1:.5f}')
                # 只存取较好记录
                if f1 > best_f1:
                    # 更新记录
                    best_f1 = f1
                    print(f'==============================================save best model ' \
                          f'best performerence {best_f1:5f}')
                    # 保存模型
                    paddle.save(model.state_dict(),
                                '{}/best.pdparams'.format(args.checkpoints))
            # 第二个轮次的迭代
            step += 1

    # 最后一个模型的保存，无论模型好坏
    if rank == 0:
        paddle.save(model.state_dict(),
                    '{}/final.pdparams'.format(args.checkpoints))


def do_predict():
    # 设置程序运行设备
    paddle.set_device(args.device)

    # 词表tokenizer
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    # 加载标签词典
    label_map = load_dict(args.tag_path)
    # 翻转标签词典
    id2label = {val: key for key, val in label_map.items()}
    # 加载预训练模型
    model = ErnieForTokenClassification.from_pretrained(
        "ernie-1.0", num_classes=len(label_map))

    # 没有实体标签的用O表示
    no_entity_label = "O"
    # 就散标签的长度
    ignore_label = len(label_map)

    print("============start predict==========")
    # 判断是否有初始化文件夹，及判断是否是文件
    if not args.init_ckpt or not os.path.isfile(args.init_ckpt):
        raise Exception("init checkpoints {} not exist".format(args.init_ckpt))
    else:
        # 加载静态模型
        state_dict = paddle.load(args.init_ckpt)
        # 模型填充参数
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.init_ckpt)

    # load data from predict file
    # 按行读取需要预测的数据
    sentences = read_by_lines(args.predict_data)  # origin data format
    # 句子灭一行是一个json,转换成长列表内嵌json
    sentences = [json.loads(sent) for sent in sentences]

    # 编码输入的列表
    encoded_inputs_list = []
    # 便利每一句
    for sent in sentences:
        # 替换掉每一句空格
        sent = sent["text"].replace(" ", "\002")
        # 转换句子为对应的文字id,类型id,和序列长度
        input_ids, token_type_ids, seq_len = convert_example_to_feature(
            [list(sent), []],
            tokenizer,
            max_seq_len=args.max_seq_len,
            is_test=True)
        # 拼接到预测数据集列表上
        encoded_inputs_list.append((input_ids, token_type_ids, seq_len))

    # 构建一个数据集
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # input_ids
        Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token], dtype='int32'
            ),  # token_type_ids
        Stack(dtype='int64')  # sequence lens
    ): fn(samples)
    # 按照批次大小进行分割
    batch_encoded_inputs = [
        encoded_inputs_list[i:i + args.batch_size]
        for i in range(0, len(encoded_inputs_list), args.batch_size)
    ]
    results = []
    # 模型验证
    model.eval()
    # 遍历每一批的数据
    for batch in batch_encoded_inputs:
        # 获取一个批次数据
        input_ids, token_type_ids, seq_lens = (batch)
        # 对token id转张量
        input_ids = paddle.to_tensor(input_ids)
        # 对token type ID转张量
        token_type_ids = paddle.to_tensor(token_type_ids)
        # 模型进行计算
        logits = model(input_ids, token_type_ids)
        # 对linear层结果进行softmax归一化，方便计算每一个类别的概率，是在最后一个轴上的，第一个轴默认是批次
        probs = F.softmax(logits, axis=-1)
        # 获取最后一维尚，数据最大的下标
        probs_ids = paddle.argmax(probs, -1).numpy()
        # 将概率转换成概率
        probs = probs.numpy()
        # 按照计算的概率和以及，每一个位置类型id，和序列长度，遍历
        for p_list, p_ids, seq_len in zip(probs.tolist(), probs_ids.tolist(),
                                          seq_lens.tolist()):
            prob_one = [
                p_list[index][pid]  # 每个字符对应的类别
                for index, pid in enumerate(p_ids[1:seq_len - 1])
            ]
            # 根据有效序列，切片有效字符，并查询对应的标签
            label_one = [id2label[pid] for pid in p_ids[1:seq_len - 1]]
            # 把结果，以及对应的标签组合起来
            results.append({"probs": prob_one, "labels": label_one})
    # 断言 句子长度和结果长度是否一致
    assert len(results) == len(sentences)
    # 遍历对应句子以及预测结果
    for sent, ret in zip(sentences, results):
        sent["pred"] = ret

    # 将结果转换层json类型，并拼接成列表
    sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
    # 按行写到本地
    write_by_lines(args.predict_save_path, sentences)
    # 打印已处理句子长度，以及结果保存的路径
    print("save data {} to {}".format(len(sentences), args.predict_save_path))


if __name__ == '__main__':
    # 判断do_train参数是否为True
    if args.do_train:
        print("train")
        do_train()
    elif args.do_predict:
        print("predict")
        do_predict()
