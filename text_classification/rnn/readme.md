

# 1. 项目简介
## 1.1. 目录结构
```buildoutcfg
rnn/
├── deploy # 部署
│   └── python
│       └── predict.py # python预测部署示例
├── export_model.py # 动态图参数导出静态图参数脚本
├── model.py # 模型组网脚本
├── predict.py # 模型预测
├── utils.py # 数据处理工具
├── train.py # 训练模型主程序入口，包括训练、评估
└── README.md # 文档说明
```

## 1.2. 模型简介
| 模型                                             | 模型介绍                                                     |
| ------------------------------------------------ | ------------------------------------------------------------ |
| BOW（Bag Of Words）                              | 非序列模型，将句子表示为其所包含词的向量的加和               |
| RNN (Recurrent Neural Network)                   | 序列模型，能够有效地处理序列信息                             |
| GRU（Gated Recurrent Unit）                      | 序列模型，能够较好地解决序列文本中长距离依赖的问题           |
| LSTM（Long Short Term Memory）                   | 序列模型，能够较好地解决序列文本中长距离依赖的问题           |
| Bi-LSTM（Bidirectional Long Short Term Memory）  | 序列模型，采用双向LSTM结构，更好地捕获句子中的语义特征       |
| Bi-GRU（Bidirectional Gated Recurrent Unit）     | 序列模型，采用双向GRU结构，更好地捕获句子中的语义特征        |
| Bi-RNN（Bidirectional Recurrent Neural Network） | 序列模型，采用双向RNN结构，更好地捕获句子中的语义特征        |
| Bi-LSTM Attention                                | 序列模型，在双向LSTM结构之上加入Attention机制，结合上下文更好地表征句子语义特征 |
| TextCNN                                          | 序列模型，使用多种卷积核大小，提取局部区域地特征             |

## 1.3. 性能评测
| 模型              | dev acc | test acc |
| ----------------- | ------- | -------- |
| BoW               | 0.8970  | 0.8908   |
| Bi-LSTM           | 0.9098  | 0.8983   |
| Bi-GRU            | 0.9014  | 0.8785   |
| Bi-RNN            | 0.8649  | 0.8504   |
| Bi-LSTM Attention | 0.8992  | 0.8856   |
| TextCNN           | 0.9102  | 0.9107   |



# 2. 操作指南
## 2.1. 训练阶段
### 2.1.1. CPU
```buildoutcfg
python train.py --vocab_path='./vocab.json' \
    --device=cpu \
    --network=bilstm \
    --lr=5e-4 \
    --batch_size=64 \
    --epochs=10 \
    --save_dir='./checkpoints'
```

### 2.1.2. GPU
```buildoutcfg
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" train.py \
    --vocab_path='./vocab.json' \
    --device=gpu \
    --network=bilstm \
    --lr=5e-4 \
    --batch_size=64 \
    --epochs=10 \
    --save_dir='./checkpoints'
```

### 2.2. 预测阶段
### 2.2.1. CPU
```
python predict.py --vocab_path='output/vocab.json' \
    --device=cpu \
    --network=bilstm \
    --params_path=checkpoints/final.pdparams
```

### 2.2.2. GPU 
```
export CUDA_VISIBLE_DEVICES=0
python predict.py --vocab_path='output/vocab.json' \
    --device=gpu \
    --network=bilstm \
    --params_path='checkpoints/final.pdparams'
```



# 3. 参考链接
- https://canvas.stanford.edu/files/1090785/download
- https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- https://arxiv.org/abs/1412.3555
- https://arxiv.org/pdf/1506.00019
- https://arxiv.org/abs/1404.2188