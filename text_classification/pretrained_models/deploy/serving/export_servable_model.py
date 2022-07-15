#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-07-11 22:27
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    export_servable_model.py
# @Project: pretrained_models
# @Package: 
# @Ref:

# limitations under the License.

import argparse
import paddle
import paddle_serving_client.io as serving_io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_model_dir",
                        type=str,
                        default="./export/",
                        help="The directory of the inference model.")
    parser.add_argument("--model_file",
                        type=str,
                        default='inference.pdmodel',
                        help="The inference model file name.")
    parser.add_argument("--params_file",
                        type=str,
                        default='inference.pdiparams',
                        help="The input inference parameters file name.")
    return parser.parse_args()


if __name__ == '__main__':
    paddle.enable_static()
    args = parse_args()
    feed_names, fetch_names = serving_io.inference_model_to_serving(
        dirname=args.inference_model_dir,
        serving_server="serving_server",
        serving_client="serving_client",
        model_filename=args.model_file,
        params_filename=args.params_file)
    print("model feed_names : %s" % feed_names)
    print("model fetch_names : %s" % fetch_names)
