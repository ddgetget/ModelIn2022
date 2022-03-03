#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-03-03 11:49
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    ModelApplication.py
# @Project: ModelIn2022
# @Package:

class BaseModel(object):
    def __init__(self, params={}):
        self.params = params
        self.model = None

    def save_model(self, model_path="./"):
        pass

    def load_model(self, model_path="./"):
        pass

    def fit(self, datasets):
        self.model = "model"
        self.datasets = datasets

    def transform(self, datasets):
        self.datasets = datasets

    def predict(self, data):
        result = self.model.predict(data)
        return result

    def api(self, data):
        result = {}
        result
