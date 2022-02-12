# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import sys

sys.path.append('..')
from autocomplete.gpt2_model import GPT2Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="download/train.txt", help="Train file path")
    parser.add_argument("--valid_file", type=str, default="download/valid.txt", help="Valid file path")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predict.")
    parser.add_argument("--model_dir", type=str, default="outputs-fine-tuned/", help="Model save dir")
    parser.add_argument("--num_epochs", type=int, default=5, help="Num of training epochs")
    args = parser.parse_args()
    print(args)

    model = GPT2Model(model_name_or_path="gpt2", max_seq_length=128, do_lower_case=False)
    if args.do_train:
        model.train_model(args.train_file, args.model_dir, eval_file=args.valid_file, num_epochs=args.num_epochs)
        print(f"model saved to {args.model_dir}")
    if args.do_predict:
        prompts = [
            "Despite the recent successes of deep learning, such models are still far from some human abilities like learning from few examples, reasoning and explaining decisions. In this paper, we focus on organ annotation in medical images and we introduce a reasoning framework that is based on learning fuzzy relations on a small dataset for generating explanations.",
            "There is a growing interest and literature on intrinsic motivations and open-ended learning in both cognitive robotics and machine learning on one side, ",
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
            print("\n\n======================\n\n")
            print("Query:", prompt)
            print("Result:", outputs[0])
            print("\n\n======================\n\n")
