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


parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--model_file",type=str,default="checkpoints/sta")
