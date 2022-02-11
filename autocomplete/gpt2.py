# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import torch
from simpletransformers.language_generation import LanguageGenerationModel
from simpletransformers.language_modeling import LanguageModelingModel
import transformers

transformers.logging.set_verbosity_error()
use_cuda = torch.cuda.is_available()


def predict_with_original_gpt2(prompts):
    model = LanguageGenerationModel("gpt2", "gpt2", args={"max_length": 64, "cache_dir": None}, use_cuda=use_cuda)
    # "cache_dir": None means use default cache dir: ~/.cache/huggingface/transformers/

    for prompt in prompts:
        # Generate text using the model. Verbose set to False to prevent logging generated sequences.
        generated = model.generate(prompt, verbose=False)

        generated = generated[0]
        print("=============================================================================")
        print(generated)
        print("=============================================================================")


def train(model_dir="outputs/fine-tuned/", train_file="download/train.txt", valid_file="download/valid.txt",
          num_train_epochs=3):
    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "block_size": 512,
        "max_seq_length": 64,
        "learning_rate": 5e-6,
        "train_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": num_train_epochs,
        "mlm": False,
        "cache_dir": os.path.expanduser("~/.cache/huggingface/transformers/"),
        "output_dir": model_dir,
        "dataset_type": "text",
    }

    model = LanguageModelingModel("gpt2", "gpt2", args=train_args, use_cuda=use_cuda)
    model.train_model(train_file, eval_file=valid_file)
    print(f"model saved to {model_dir}")
    print(model.eval_model(valid_file))


class Infer:
    def __init__(self, model_name="gpt2", model_dir="outputs/fine-tuned", use_cuda=use_cuda, max_length=64):
        self.model_name = model_name
        args = {"max_length": max_length, "cache_dir": None}
        # cache_dir: None means use default cache dir: ~/.cache/huggingface/transformers/
        self.model = LanguageGenerationModel(model_name, model_dir, args=args, use_cuda=use_cuda)

    def predict(self, prompt):
        """
        Generate text using the model. Verbose set to False to prevent logging generated sequences.
        :param prompt: str, input string
        :return: str
        """
        generated = self.model.generate(prompt, verbose=False)
        generated = generated[0]
        return generated


if __name__ == '__main__':
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

    predict_with_original_gpt2(prompts)
    train(model_dir="outputs/fine-tuned/", num_train_epochs=3)
    infer = Infer(model_name="gpt2", model_dir="outputs/fine-tuned", use_cuda=use_cuda)
    for p in prompts:
        r = infer.predict(p)
        print(r)
