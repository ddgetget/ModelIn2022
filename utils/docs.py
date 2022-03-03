#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time:    2022-03-03 12:57
# @Author:  geng
# @Email:   yonglonggeng@163.com
# @WeChat:  superior_god
# @File:    docs.py
# @Project: ModelIn2022
# @Package:
class ReporInfot():
    def __init__(self):
        self.lines = []

    def w_version(self, data):
        self.lines.append("#### 基本信息\r\n")
        for key, value in data.items():
            line = "**{}**:{}\r\n".format(key, value)
            self.lines.append(line)

    def write(self):
        with open("outputs/report.md", "w", encoding='utf-8') as f:
            f.writelines(self.lines)


if __name__ == '__main__':
    data = {'auther': 'LongGengYung', 'version': 'V-1', 'date': '2022-3-3 12:09:43', 'task': 'CV',
            'application': 'ImageClassificationFromScratch'}
    # lines = []
    # lines.append("#### 基本信息\r\n")
    # for key, value in data.items():
    #     line = "**{}**:{}\r\n".format(key, value)
    #     lines.append(line)
    # print(lines)
    # with open("../outputs/report1.md", "w", encoding='utf-8') as f:
    #     f.writelines(lines)
    report = ReporInfot()
    report.w_version(data)
    report.write()
