# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import sys
import torch

sys.path.append('..')
from autocomplete.gpt2 import Infer

use_cuda = torch.cuda.is_available()
m = Infer(model_name="gpt2", model_dir="shibing624/code-autocomplete-gpt2-base", use_cuda=use_cuda)
i = m.predict('import torch.nn as')
print(i)
