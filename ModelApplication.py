#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-03-03 11:49
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    ModelApplication.py
# @Project: ModelIn2022
# @Package:
from utils.Version import VersionInfo
from utils.docs import ReporInfot


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
        return result

    def describe(self, info, export=False):
        versionInfo = VersionInfo(info=info)
        desc = {}
        for item in versionInfo.__dict__.items():
            if item[0] == "info":
                continue
            desc[item[0]] = item[1]

        if export:
            report = ReporInfot()
            report.w_version(desc)
            report.write()
        return desc
