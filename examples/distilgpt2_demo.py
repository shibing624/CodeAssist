# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from codeassist import GPT2Coder

m = GPT2Coder("shibing624/code-autocomplete-distilgpt2-python")
print(m.generate('def load_csv_file(file_path):')[0])
