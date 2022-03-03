#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-03-03 11:54
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    Version.py
# @Project: ModelIn2022
# @Package:
import datetime


class VersionInfo(object):
    def __init__(self, info={}):
        self.info = info
        self.auther = "LongGengYung"
        self.version = "V-1"
        self.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.task = "CV"
        self.application = "ImageClassificationFromScratch"
        self.function = "your module function"


    def update_version(self):
        self.version = self.info['version'] if "version" in self.info.keys() else self.version
        self.task = self.info['task'] if "date" in self.info.keys() else self.task
        self.application = self.info['application'] if "date" in self.info.keys() else self.application
        self.function = self.info['function'] if "date" in self.info.keys() else self.function


if __name__ == '__main__':
    info = {"version": "V-2.0"}
    versionInfo = VersionInfo(info=info)
    versionInfo.update_version()
    print(versionInfo.date)
    print(versionInfo)
