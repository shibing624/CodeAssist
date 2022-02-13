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
        """from torch import nn
        class LSTM(Module):
            def __init__(self, *,
                         n_tokens: int,
                         embedding_size: int,
                         hidden_size: int,
                         n_layers: int):""",
        """import numpy as np
        import torch
        import torch.nn as""",
        "import java.util.ArrayList;",
    ]

    m = GPT2Coder("shibing624/code-autocomplete-gpt2-base")
    for i in m.generate('import torch.nn as', num_return_sequences=3):
        print(i)

    for prompt in prompts:
        decoded = m.generate(prompt, num_return_sequences=1)
        print("Input :", prompt)
        print("Output:", decoded[0])
        print("=" * 20)
