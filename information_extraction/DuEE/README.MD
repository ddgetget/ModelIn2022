

# 1. 项目概况
## 1.1. 项目简介

<div align="center">
<img src="DuEE-Fin/ee.png" width="600" height="200" alt="事件抽取" align=center />
</div>

## 1.2. 项目目录
```buildoutcfg
DuEE/
├── classifier.py # 文本分类训练脚本
├── duee_1_data_prepare.py # 句子级事件抽取数据预处理
├── duee_1_postprocess.py # 句子级事件抽取数据后处理
├── duee_fin_data_prepare.py  # 篇章级事件抽取数据预处理
├── duee_fin_postprocess.py  # 篇章级事件抽取数据后处理
├── README.md # 文档说明
├── run_classifier.sh # 文本分类训练启动脚本
├── run_duee_1.sh # 句子级事件抽取启动脚本
├── run_duee_fin.sh # 篇章级事件抽取启动脚本
├── run_sequence_labeling.sh # 序列标注训练启动脚本
├── sequence_labeling.py # 序列标注训练脚本
└── utils.py # 效能函数
```


# 2. 篇章级事件抽取基线
## 2.1. 操作步骤
### 2.1.1. 数据处理

```buildoutcfg
dataset_name=DuEE-Fin
data_dir=/Users/geng/Documents/data/${dataset_name}
conf_dir=conf/${dataset_name}
ckpt_dir=checkpoints/${dataset_name}
submit_data_path=outputs/test_duee_fin.json
pred_data=${data_dir}/test.json  # 换其他数据，需要修改它

learning_rate=5e-5
max_seq_len=300
batch_size=16
epoch=20

python duee_fin_data_prepare.py
```

### 2.1.2. 触发词训练

```
dataset_name=DuEE-Fin
data_dir=/Users/geng/Documents/data/DuEE-fin
conf_dir=conf/DuEE-fin
ckpt_dir=checkpoints/${dataset_name}
submit_data_path=outputs/test_duee_fin.json
pred_data=/Users/geng/Documents/data/DuEE-fin/test.json  # 换其他数据，需要修改它

learning_rate=5e-5
max_seq_len=300
batch_size=16
epoch=20

model=trigger 
is_train=True
pred_save_path=${ckpt_dir}/${model}/test_pred.json
sh run_sequence_labeling.sh ${data_dir}/${model} ${conf_dir}/${model}_tag.dict ${ckpt_dir}/${model} ${pred_data} ${learning_rate} ${is_train} ${max_seq_len} ${batch_size} ${epoch} ${pred_save_path}

```
### 2.1.3. 触发词预测

```
dataset_name=DuEE-Fin
data_dir=/Users/geng/Documents/data/${dataset_name}
conf_dir=conf/${dataset_name}
ckpt_dir=checkpoints/${dataset_name}
submit_data_path=outputs/test_duee_fin.json
pred_data=${data_dir}/test.json  # 换其他数据，需要修改它

learning_rate=5e-5
max_seq_len=300
batch_size=16
epoch=20

model=trigger 
is_train=False
pred_save_path=${ckpt_dir}/${model}/test_pred.json
sh run_sequence_labeling.sh ${data_dir}/${model} ${conf_dir}/${model}_tag.dict ${ckpt_dir}/${model} ${pred_data} ${learning_rate} ${is_train} ${max_seq_len} ${batch_size} ${epoch} ${pred_save_path}
```

### 2.1.4. 论元训练

```
dataset_name=DuEE-Fin
data_dir=/Users/geng/Documents/data/${dataset_name}
conf_dir=conf/${dataset_name}
ckpt_dir=checkpoints/${dataset_name}
submit_data_path=outputs/test_duee_fin.json
pred_data=${data_dir}/test.json  # 换其他数据，需要修改它

learning_rate=5e-5
max_seq_len=300
batch_size=16
epoch=20

model=role 
is_train=True
pred_save_path=${ckpt_dir}/${model}/test_pred.json
sh run_sequence_labeling.sh ${data_dir}/${model} ${conf_dir}/${model}_tag.dict ${ckpt_dir}/${model} ${pred_data} ${learning_rate} ${is_train} ${max_seq_len} ${batch_size} ${epoch} ${pred_save_path}
```


### 2.1.5. 论元预测

```
dataset_name=DuEE-Fin
data_dir=/Users/geng/Documents/data/${dataset_name}
conf_dir=conf/${dataset_name}
ckpt_dir=checkpoints/${dataset_name}
submit_data_path=outputs/test_duee_fin.json
pred_data=${data_dir}/test.json  # 换其他数据，需要修改它

learning_rate=5e-5
max_seq_len=300
batch_size=16
epoch=20

model=role 
is_train=False
pred_save_path=${ckpt_dir}/${model}/test_pred.json
sh run_sequence_labeling.sh ${data_dir}/${model} ${conf_dir}/${model}_tag.dict ${ckpt_dir}/${model} ${pred_data} ${learning_rate} ${is_train} ${max_seq_len} ${batch_size} ${epoch} ${pred_save_path}
```

### 2.1.6. 枚举训练

```
dataset_name=DuEE-Fin
data_dir=/Users/geng/Documents/data/${dataset_name}
conf_dir=conf/${dataset_name}
ckpt_dir=checkpoints/${dataset_name}
submit_data_path=outputs/test_duee_fin.json
pred_data=${data_dir}/test.json  # 换其他数据，需要修改它

learning_rate=5e-5
max_seq_len=300
batch_size=16
epoch=20

model=enum
is_train=True
pred_save_path=${ckpt_dir}/${model}/test_pred.json
sh run_classifier.sh ${data_dir}/${model} ${conf_dir}/${model}_tag.dict ${ckpt_dir}/${model} ${pred_data} ${learning_rate} ${is_train} ${max_seq_len} ${batch_size} ${epoch} ${pred_save_path}
}
```

### 2.1.7. 枚举预测

```
dataset_name=DuEE-Fin
data_dir=/Users/geng/Documents/data/${dataset_name}
conf_dir=conf/${dataset_name}
ckpt_dir=checkpoints/${dataset_name}
submit_data_path=outputs/test_duee_fin.json
pred_data=${data_dir}/test.json  # 换其他数据，需要修改它

learning_rate=5e-5
max_seq_len=300
batch_size=16
epoch=20

model=enum
is_train=False
pred_save_path=${ckpt_dir}/${model}/test_pred.json
sh run_classifier.sh ${data_dir}/${model} ${conf_dir}/${model}_tag.dict ${ckpt_dir}/${model} ${pred_data} ${learning_rate} ${is_train} ${max_seq_len} ${batch_size} ${epoch} ${pred_save_path}
}
```



## 2.1. 数据预处理并加载
篇章级事件抽取数据集（DuEE-Fin）是金融领域篇章级别事件抽取数据集， 共包含13个已定义好的事件类型约束和1.15万中文篇章（存在部分非目标篇章作为负样例），其中6900训练集，1150验证集和3450测试集，数据集下载地址 。 在该数据集上基线采用基于ERNIE的序列标注（sequence labeling）方案，分为基于序列标注的触发词抽取模型、基于序列标注的论元抽取模型和枚举属性分类模型，属于PipeLine模型；基于序列标注的触发词抽取模型采用BIO方式，识别触发词的位置以及对应的事件类型，基于序列标注的论元抽取模型采用BIO方式识别出事件中的论元以及对应的论元角色；枚举属性分类模型采用ernie进行分类。

```buildoutcfg
sh run_duee_fin.sh data_prepare
```


## 2.2 定义损失函数和优化器，开始训练
```buildoutcfg
# 触发词识别模型训练
sh run_duee_fin.sh trigger_train

# 触发词识别预测
sh run_duee_fin.sh trigger_predict

# 论元识别模型训练
sh run_duee_fin.sh role_train

# 论元识别预测
sh run_duee_fin.sh role_predict

# 枚举分类模型训练
sh run_duee_fin.sh enum_train

# 枚举分类预测
sh run_duee_fin.sh enum_predict
```
## 2.3. 数据后处理，提交结果
```buildoutcfg
sh run_duee_fin.sh pred_2_submit
```

# 3. 句子级事件抽取基线

## 3.1. 训练
```buildoutcfg
# 训练触发词识别模型
sh run_duee_1.sh trigger_train

# 触发词识别预测
sh run_duee_1.sh trigger_predict

# 论元识别模型训练
sh run_duee_1.sh role_train

# 论元识别预测
sh run_duee_1.sh role_predict

# 数据后处理，提交预测结果
# 结果存放于submit/test_duee_1.json`
sh run_duee_1.sh pred_2_submit
```

# 4. 原理概述

# 5. 参考资料

[DuEE: A Large-Scale Dataset for Chinese Event Extraction in Real-World Scenarios](https://link.springer.com/chapter/10.1007/978-3-030-60457-8_44)