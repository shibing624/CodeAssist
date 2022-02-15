# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

sys.path.append('..')
from autocomplete.gpt2_coder import GPT2Coder

if __name__ == '__main__':
    prompts = [
        "import numpy as np",
        "import torch.nn as",
        'parser.add_argument("--num_train_epochs",',
        "def set_seed(",
        "def factorial",
    ]

    m = GPT2Coder("gpt2")
    for i in m.generate('import torch.nn as', num_return_sequences=3):
        print(i)

    for prompt in prompts:
        decoded = m.generate(prompt, num_return_sequences=1)
        print("Input :", prompt)
        print("Output:", decoded[0])
        print("=" * 20)
