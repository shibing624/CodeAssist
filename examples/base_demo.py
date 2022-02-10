# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

sys.path.append('..')
from autocomplete.gpt2 import predict

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
    for prompt in prompts:
        res = predict(prompt, model_dir='shibing624/code-autocomplete-gpt2')
        print("\n\n======================\n\n")
        print("Query:", prompt)
        print("\nResult:, res")
        print("\n\n======================\n\n")
