#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-03-03 11:54
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    Version.py
# @Project: ModelIn2022
# @Package:
class VersionInfo(object):
    def __init__(self, params={}):
        self.params = params
        self.auther = "LongGengYung"
        self.version = "V-1"
        self.date = "2022-3-3 12:09:43"
        self.task = "CV"
        self.application = "ImageClassificationFromScratch"

    def update_version(self):
        self.version = self.params['version'] if "version" in self.params.keys() else self.version
        self.date = self.params['date'] if "date" in self.params.keys() else self.date
        self.task = self.params['task'] if "date" in self.params.keys() else self.task
        self.application = self.params['application'] if "date" in self.params.keys() else self.application


if __name__ == '__main__':
    params = {"version": "V-2.0"}
    versionInfo = VersionInfo(params=params)
    versionInfo.update_version()
    print(versionInfo.date)
    print(versionInfo)
