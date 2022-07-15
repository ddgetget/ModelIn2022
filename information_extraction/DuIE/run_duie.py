#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-12 21:39
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    run_duie.py
# @Project: DuIE
# @Package: 
# @Ref:


import argparse
import json
import os
import random
import sys
import time

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from data_loader import DuIEDataset, DataCollator
from paddle.io import DataLoader
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification, LinearDecayWithWarmup
from tqdm import tqdm

from utils import decoding, get_precision_recall_f1, write_prediction_results

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action='store_true', default=False, help="训练触发器")
parser.add_argument("--do_predict", action='store_true', default=False, help="预测触发器")
parser.add_argument("--init_checkpoint", default="checkpoints", type=str, required=False, help="预测部分模型路径")
parser.add_argument("--data_path", default="/Users/geng/data/DuIE2.0/", type=str, required=False, help="训练数据根目录")
parser.add_argument("--predict_data_file", default="/Users/geng/data/DuIE2.0/duie_test2.json/duie_test2.json", type=str,
                    required=False,
                    help="预测用到的数据")
parser.add_argument("--output_dir", default="checkpoints", type=str, required=False,
                    help="模型输出文件夹")
parser.add_argument("--max_seq_length", default=128, type=int, help="最大可处理字符串长度")
parser.add_argument("--batch_size", default=8, type=int, help="每一次训练的大小，即批大小", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="学习率")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="衰减系数，是一个float类型或者shape为[1] ，数据类型为float32的Tensor类型。默认值为0.01。")
parser.add_argument("--num_train_epochs", default=3, type=int, help="模型训练次数")
parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--seed", default=42, type=int, help="随机数种子")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="cpu",
                    help="选择需要运行的平台")
args = parser.parse_args()


# yapf: enable


class BCELossForDuIE(nn.Layer):

    def __init__(self, ):
        super(BCELossForDuIE, self).__init__()
        # 返回计算BCEWithLogitsLoss的可调用对象。
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask):
        # 计算标签和逻辑值之间的损失
        loss = self.criterion(logits, labels)
        # 构建掩码
        mask = paddle.cast(mask, 'float32')
        # 对部分loss删除
        loss = loss * mask.unsqueeze(-1)
        # 在2轴上求平均值，在1轴上求和，在基于1轴求均值
        loss = paddle.sum(loss.mean(axis=2), axis=1) / paddle.sum(mask, axis=1)
        # 最后在求总的均值，即总的loss
        loss = loss.mean()
        # 返回loss
        return loss


def set_random_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, data_loader, file_path, mode):
    """
    mode eval:
    eval on development set and compute P/R/F1, called between training.
    mode predict:
    eval on development / test set, then write predictions to \
        predict_test.json and predict_test.json.zip \
        under args.data_path dir for later submission or evaluation.
    """
    example_all = []
    # 打开需要性预测的文件
    with open(file_path, "r", encoding="utf-8") as fp:
        # 按行读取数据
        for line in fp:
            # 结果拼接到熬列表中
            example_all.append(json.loads(line))
    # 获取id2spo的文件路径
    id2spo_path = os.path.join(os.path.dirname(file_path), "id2spo.json")
    # 读取文件
    with open(id2spo_path, 'r', encoding='utf8') as fp:
        # 加载成字典
        id2spo = json.load(fp)

    # 模型验证
    model.eval()
    # 设置损失率为0
    loss_all = 0
    # 设置验证步数为0
    eval_steps = 0
    # 结果输出列表
    formatted_outputs = []
    # 当前id
    current_idx = 0
    # 遍历data_loader这个迭代器
    for batch in tqdm(data_loader, total=len(data_loader)):
        # 每迭代一次，跟新计数器
        eval_steps += 1
        # 获取一个批次的数据
        input_ids, seq_len, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
        # 对数据进行计算，获取逻辑结果
        logits = model(input_ids=input_ids)
        # TODO 暂时无法理解
        mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and(
            (input_ids != 2))
        # 结果与标签进行对比就散，获取损失值
        loss = criterion(logits, labels, mask)
        # 所有损失和，numpy的item函数用于返回指定索引处的数据
        loss_all += loss.numpy().item()
        # 对逻辑结果，引入非线性因素。
        probs = F.sigmoid(logits)
        # 对概率数字类型转换
        logits_batch = probs.numpy()
        # 获取序列长度
        seq_len_batch = seq_len.numpy()
        # 获取其实索引
        tok_to_orig_start_index_batch = tok_to_orig_start_index.numpy()
        # 获取尾部索引
        tok_to_orig_end_index_batch = tok_to_orig_end_index.numpy()
        # 结果拼接
        formatted_outputs.extend(
            decoding(example_all[current_idx:current_idx + len(logits)], id2spo,
                     logits_batch, seq_len_batch, tok_to_orig_start_index_batch,
                     tok_to_orig_end_index_batch))
        # 更新之前的id
        current_idx = current_idx + len(logits)
    # 求平均损失
    loss_avg = loss_all / eval_steps
    print("eval loss: %f" % (loss_avg))

    if mode == "predict":
        # 针对预测的，写入到预测文件
        predict_file_path = os.path.join(args.data_path, 'predictions.json')
    else:
        # 否则写到验证文件
        predict_file_path = os.path.join(args.data_path, 'predict_eval.json')

    # 写到本地
    predict_zipfile_path = write_prediction_results(formatted_outputs,
                                                    predict_file_path)

    if mode == "eval":
        # 获取准确率，召回率，和F1-Score
        precision, recall, f1 = get_precision_recall_f1(file_path,
                                                        predict_zipfile_path)
        # 针对验证集，直接把测试运行结果删了，作者想的法子奴婢了
        os.system('rm {} {}'.format(predict_file_path, predict_zipfile_path))
        # 返回准确率，召回率，和F1-score
        return precision, recall, f1
    elif mode != "predict":
        raise Exception("wrong mode for eval func")


def do_train():
    # 设置程序运行的设备
    paddle.set_device(args.device)
    # 获取当前进行进程的rank
    rank = paddle.distributed.get_rank()
    # 获取当前任务的进程数
    if paddle.distributed.get_world_size() > 1:
        # 当前进程数等于环境变量 PADDLE_TRAINERS_NUM 的值，默认值为1。
        paddle.distributed.init_parallel_env()

    # 拼接关系ID字典路径
    label_map_path = os.path.join(args.data_path, "predicate2id.json")
    # 判断路径是否存在
    if not (os.path.exists(label_map_path) and os.path.isfile(label_map_path)):
        sys.exit("{} dose not exists or is not a file.".format(label_map_path))

    # 打开关系ID字典文件
    with open(label_map_path, 'r', encoding='utf8') as fp:
        # 读取本地json文件到字典
        label_map = json.load(fp)

    # 获取字典的大小，并计算对应的标签个数，出去I，O标签，处理头实体/尾实体，计算公式为2*N+2，
    num_classes = (len(label_map.keys()) - 2) * 2 + 2

    # 加载预训练模型
    model = ErnieForTokenClassification.from_pretrained("ernie-1.0",
                                                        num_classes=num_classes)
    # 通过数据并行模式执行动态图模型。
    model = paddle.DataParallel(model)
    # 加载对应的词表
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    # 调用损失函数
    criterion = BCELossForDuIE()

    # 加载训练数据集(data, label_map, tokenizer, max_length,
    #                        pad_to_max_length)
    train_dataset = DuIEDataset.from_file(
        os.path.join(args.data_path, 'train_data.json'), tokenizer,
        args.max_seq_length, True)
    # 分布式批采样器加载数据的一个子集
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    collator = DataCollator()
    # 训练数据加载，返回一个迭代器
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=collator,
                                   return_list=True)
    # 汇编验证数据路径
    eval_file_path = os.path.join(args.data_path, 'dev_data.json')
    # 读取本地文件
    test_dataset = DuIEDataset.from_file(eval_file_path, tokenizer,
                                         args.max_seq_length, True)
    # 批采样器的基础实现，用于 paddle.io.DataLoader 中迭代式获取mini-batch的样本下标数组，数组长度与 batch_size 一致。
    test_batch_sampler = paddle.io.BatchSampler(test_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                drop_last=True)
    # 返回一个迭代器
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_sampler=test_batch_sampler,
                                  collate_fn=collator,
                                  return_list=True)

    # Defines learning rate strategy.
    # 计算训练集大小
    steps_by_epoch = len(train_data_loader)
    # 计算训练总步数
    num_training_steps = steps_by_epoch * args.num_train_epochs
    # 动态学习率
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_ratio)
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    # 参数，只针对训练的数据
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    # AdamW优化器出自 DECOUPLED WEIGHT DECAY REGULARIZATION，用来解决 Adam 优化器中L2正则化失效的问题。
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    # Starts training.
    # 全局迭代次数
    global_step = 0
    # 日志步数
    logging_steps = 50
    # 存储模型步数
    save_steps = 10000
    # 记录开始时间
    tic_train = time.time()
    # 按照迭代次数遍历
    for epoch in range(args.num_train_epochs):
        print("\n=====start training of %d epochs=====" % epoch)
        # 记录总次数时间
        tic_epoch = time.time()
        # 模型训练
        model.train()
        # 枚举迭代训练集
        for step, batch in enumerate(train_data_loader):
            # 获取对应的一个批次的数据，包括输入文字的id,序列的长度，标签其实索引，标签结束索引，标签（只有0和1）
            input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
            # 计算最后一层逻辑结果
            logits = model(input_ids=input_ids)
            # TODO 目前不太懂
            mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and(
                (input_ids != 2))
            # 交叉熵计算损失值
            loss = criterion(logits, labels, mask)
            # 损失函数反响求导
            loss.backward()
            # 优化器优化
            optimizer.step()
            # 使用有计划的学习率进行
            lr_scheduler.step()
            # 反响求导清空之前的梯度
            optimizer.clear_grad()
            # 获取损失值的值（按索引号）
            loss_item = loss.numpy().item()
            # 全局训练步数+1
            global_step += 1

            if global_step % logging_steps == 0 and rank == 0:
                # 打印训练日志
                print(
                    "epoch: %d / %d, steps: %d / %d, loss: %f, speed: %.2f step/s"
                    % (epoch, args.num_train_epochs, step, steps_by_epoch,
                       loss_item, logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % save_steps == 0 and rank == 0:
                # 打印保存模型的日志
                print("\n=====start evaluating ckpt of %d steps=====" %
                      global_step)
                # 获取验证集闪的准确率，召回率，和F1-score
                precision, recall, f1 = evaluate(model, criterion,
                                                 test_data_loader,
                                                 eval_file_path, "eval")
                # 打印准确率，召回率和F1-score
                print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
                      (100 * precision, 100 * recall, 100 * f1))
                # 打印模型保存路径
                print("saving checkpoing model_%d.pdparams to %s " %
                      (global_step, args.output_dir))
                # 模型保存
                paddle.save(
                    model.state_dict(),
                    os.path.join(args.output_dir,
                                 "model_%d.pdparams" % global_step))
                # 返回到O型训练模式
                model.train()  # back to train mode
        # 计算发费时间
        tic_epoch = time.time() - tic_epoch
        # 打印时间
        print("epoch time footprint: %d hour %d min %d sec" %
              (tic_epoch // 3600, (tic_epoch % 3600) // 60, tic_epoch % 60))

    # Does final evaluation.
    # 针对当前进程的rank==0
    if rank == 0:
        # 打印表头
        print("\n=====start evaluating last ckpt of %d steps=====" %
              global_step)
        # 计算验证指标
        precision, recall, f1 = evaluate(model, criterion, test_data_loader,
                                         eval_file_path, "eval")
        # 打印验证指标
        print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
              (100 * precision, 100 * recall, 100 * f1))
        # 模型保存
        paddle.save(
            model.state_dict(),
            os.path.join(args.output_dir, "model_%d.pdparams" % global_step))
        print("\n=====training complete=====")


def do_predict():
    # 设置系统运行环境
    paddle.set_device(args.device)

    # Reads label_map.
    # 加载关系对应的字典
    label_map_path = os.path.join(args.data_path, "predicate2id.json")
    # 真滴查询不到字典的抛出异常
    if not (os.path.exists(label_map_path) and os.path.isfile(label_map_path)):
        sys.exit("{} dose not exists or is not a file.".format(label_map_path))
    # 打开寄送文件
    with open(label_map_path, 'r', encoding='utf8') as fp:
        # 读取成字典
        label_map = json.load(fp)
    # 计算字典里面最后形成的标签个数
    num_classes = (len(label_map.keys()) - 2) * 2 + 2

    # Loads pretrained model ERNIE
    # 加载预训练模型
    model = ErnieForTokenClassification.from_pretrained("ernie-1.0",
                                                        num_classes=num_classes)
    # 加载预训练词表
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    # 定义损失函数
    criterion = BCELossForDuIE()

    # Loads dataset.
    # 加载测试数据
    test_dataset = DuIEDataset.from_file(args.predict_data_file, tokenizer,
                                         args.max_seq_length, True)
    # 数据处理
    collator = DataCollator()
    #分布式批采样器加载数据的一个子集
    test_batch_sampler = paddle.io.BatchSampler(test_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                drop_last=True)
    # DataLoader返回一个迭代器
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_sampler=test_batch_sampler,
                                  collate_fn=collator,
                                  return_list=True)

    # Loads model parameters.
    # 寻找预测需要的模型
    if not (os.path.exists(args.init_checkpoint)
            and os.path.isfile(args.init_checkpoint)):
        # 如果没有模型，就抛出异常
        sys.exit("wrong directory: init checkpoints {} not exist".format(
            args.init_checkpoint))
    # 加载模型静态词典参数
    state_dict = paddle.load(args.init_checkpoint)
    # 给模型填充数据
    model.set_dict(state_dict)

    # Does predictions.
    print("\n=====start predicting=====")
    # 模型验证，针对测试数据积极选哪个
    evaluate(model, criterion, test_data_loader, args.predict_data_file,
             "predict")
    print("=====predicting complete=====")


if __name__ == "__main__":
    # 党训练触发器打开
    if args.do_train:
        # 调用训练函数
        do_train()
    # 否则调用预测触发器
    elif args.do_predict:
        # 调用预测函数
        do_predict()
