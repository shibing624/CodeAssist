# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append('..')
from codeassist import WizardCoder

m = WizardCoder("WizardLM/WizardCoder-15B-V1.0")
print(m.generate('def load_csv_file(file_path):')[0])

prompts = [
    "write a function to load csv file and return a dataframe",
    "give me odd numbers from 1 to 10",
    "write a function to show fibonacci sequence and return a list",
    "write a Java code to sum 1 to 10.",
    "write a python code to sum 1 to 10.",
    "write a C++ code to sum 1 to 12.",
    "写个python的快排算法",
    "生成4到400之间的随机数，用java和python写代码",
    "写java代码，从1累加到100",
    "写python代码，从1累加到10",
    "写个斐波拉契数列，返回一个列表，python代码",
    "给出所有1-20的偶数列表，C++代码",
    "给出所有1-20的奇数列表，python代码",
    # below is for language model chat
    "tell me about beijing",
    "give me a plan to NewYork city for three days trip",
    "详细介绍下南京",
    "列个南京的详细三天旅游计划",
    "失眠怎么办",
]
for prompt in prompts:
    print('input :', prompt)
    print('output:', m.generate(prompt)[0])
    print()
