# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys

sys.path.append('..')
from autocomplete.gpt2_coder import GPT2Coder

m = GPT2Coder("shibing624/code-autocomplete-gpt2-base")
print(m.generate('import torch.nn as')[0])
