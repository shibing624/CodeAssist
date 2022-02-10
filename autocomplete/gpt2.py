# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import torch
from simpletransformers.language_generation import LanguageGenerationModel
from simpletransformers.language_modeling import LanguageModelingModel

use_cuda = torch.cuda.is_available()
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
    "import java.util.ArrayList",
]


def test_gpt2():
    model = LanguageGenerationModel("gpt2", "gpt2", args={"max_length": 200, "cache_dir": None}, use_cuda=use_cuda)

    for prompt in prompts:
        # Generate text using the model. Verbose set to False to prevent logging generated sequences.
        generated = model.generate(prompt, verbose=False)

        generated = ".".join(generated[0].split(".")[:-1]) + "."
        print("=============================================================================")
        print(generated)
        print("=============================================================================")


def finetune_lm():
    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "block_size": 512,
        "max_seq_length": 64,
        "learning_rate": 5e-6,
        "train_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 1,
        "mlm": False,
        "cache_dir": os.path.expanduser("~/.cache/huggingface/transformers/"),
        "output_dir": "outputs/fine-tuned/",
    }

    model = LanguageModelingModel("gpt2", "gpt2", args=train_args, use_cuda=use_cuda)
    train_file = "download/train.txt"
    valid_file = "download/valid.txt"
    model.train_model(train_file, eval_file=valid_file)
    print(model.eval_model(valid_file))

    # Use finetuned model
    model = LanguageGenerationModel("gpt2", "outputs/fine-tuned", args={"max_length": 200}, use_cuda=use_cuda)
    for prompt in prompts:
        # Generate text using the model. Verbose set to False to prevent logging generated sequences.
        generated = model.generate(prompt, verbose=False)

        generated = ".".join(generated[0].split(".")[:-1]) + "."
        print("=============================================================================")
        print(generated)
        print("=============================================================================")


def predict(prompt, model_dir="outputs/fine-tuned"):
    model = LanguageGenerationModel("gpt2", model_dir, args={"max_length": 64}, use_cuda=use_cuda)
    # Generate text using the model. Verbose set to False to prevent logging generated sequences.
    generated = model.generate(prompt, verbose=False)
    generated = ".".join(generated[0].split(".")[:-1]) + "."
    print(generated)
    return generated


if __name__ == '__main__':
    test_gpt2()
    finetune_lm()
