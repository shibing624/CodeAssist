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
    model = GPT2Model(model_name_or_path="gpt2", max_seq_length=128, do_lower_case=False)
    model.train_model(train_file, model_dir, eval_file=valid_file)
    print(f"model saved to {model_dir}")
