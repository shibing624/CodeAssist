# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

sys.path.append('..')
from autocomplete.gpt2 import predict_with_original_gpt2

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
    predict_with_original_gpt2(prompts)
