# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys

sys.path.append('..')
from autocomplete.gpt2_model import GPT2Model

if __name__ == '__main__':
    model_dir = "gpt2-fine-tuned/"
    train_file = "download/train.txt"
    valid_file = "download/valid.txt"
    #model = GPT2Model(model_name_or_path="gpt2", max_seq_length=128, do_lower_case=False)
    #model.train_model(train_file, model_dir, eval_file=valid_file)
    #print(f"model saved to {model_dir}")
    #del model

    model = GPT2Model(model_dir)
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
    "def factorial(n):",
    ]
    for prompt in prompts:
        outputs = model.generate(prompt, bad_words=['#', 'fuck'])
        print(outputs[0])
        print("=" * 20)
