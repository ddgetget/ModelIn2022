```text			
   Copyright (c) 2022 LongGengYung Authors. All Rights Reserved
====================================================================
			项 目：【关系&事件&实体抽取】
			整理人：LongGengYung
			微 信：superior_god
			邮 箱：yonglonggeng@163.com
			公众号：庚庚体验馆
			知 乎：雍珑庚
====================================================================
```

# 1. 项目概况
## 1.1. 项目概述

四类抽取任务：实体抽取、关系抽取、事件抽取和情感抽取。
以“In 1997, Steve was excited to become the CEO of Apple.”为例，各个任务的目标结构为:

- 实体：Steve - 人物实体、Apple - 组织机构实体
- 关系：(Steve, Work For Apple)
- 事件：{类别: 就职事件, 触发词: become, 论元: [[雇主, Apple], [雇员, Steve]]}
- 情感：(exicted, become the CEO of Apple, Positive)

## 1.2. 项目目录
```text
├── config/                       # 配置文件
├── inference.py                  # 推理入口
├── process_data.py               # 比赛数据处理相关脚本
├── README.md                     # 说明文件
├── requirements.txt              # Python 依赖包文件
├── run_seq2struct.py             # Python 入口
└── uie/
    ├── evaluation                # 信息抽取代码
    └── seq2struct                # 序列到结构代码.
```

# 2. 算法原理


# 3. 操作流程

## 3.1. 训练阶段
采用 Yaml 配置文件来配置不同任务的数据来源和验证方式，详见多任务配置文件 `config/multi-task-duuie.yaml`。
本例将依据配置文件自动读取每个任务所需的训练数据进行训练，并对每个任务进行验证并汇报结果。
```commandline
python3 run_seq2struct.py                              \
  --multi_task_config config/multi-task-duuie.yaml     \
  --negative_keep 1.0                                  \
  --do_train                                           \
  --metric_for_best_model=all-task-ave                 \
  --model_name_or_path=./uie-char-small                \
  --num_train_epochs=10                                \
  --per_device_train_batch_size=32                     \
  --per_device_eval_batch_size=256                     \
  --output_dir=output/duuie_multi_task_b32_lr5e-4      \
  --logging_dir=output/duuie_multi_task_b32_lr5e-4_log \
  --learning_rate=5e-4                                 \
  --overwrite_output_dir                               \
  --gradient_accumulation_steps 1
```
## 3.2. 验证阶段


## 3.3. 测试阶段

## 3.4. 部署阶段



# 4. 项目总结
## 4.1. 项目陈述

## 4.2. 项目优势

## 4.3. 项目缺陷

## 4.4. 避坑指南

## 4.5. 注意事项


# 5. 参考信息
