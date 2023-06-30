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
    "write a function to load csv file",
    "write a function to load csv file and return a dataframe",
    "sort a list of integers",
    "give me odd numbers from 1 to 10",
    "write a function to show fibonacci sequence",
    "write a function to show fibonacci sequence and return a list",
    "write a Java code to sum 1 to 10.",
    "write a python code to sum 1 to 10.",
    "write a C++ code to sum 1 to 12.",
]
for prompt in prompts:
    print(prompt)
    print(m.generate(prompt)[0])