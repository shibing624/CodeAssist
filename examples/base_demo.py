# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import sys
import torch

sys.path.append('..')
from autocomplete.gpt2 import Infer

use_cuda = torch.cuda.is_available()
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
        "import java.util.ArrayList",
    ]
    infer = Infer(model_name="gpt2", model_dir="shibing624/code-autocomplete-gpt2-base", use_cuda=use_cuda)
    for prompt in prompts:
        res = infer.predict(prompt)
        print("Query:", prompt)
        print("Result:", res)
