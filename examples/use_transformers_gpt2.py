# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("shibing624/code-autocomplete-gpt2-base")
model = GPT2LMHeadModel.from_pretrained("shibing624/code-autocomplete-gpt2-base")
model.to(device)
prompts = [
    "def load_csv_file(file_path):",
    "import numpy as np",
    "import torch.nn as",
    'parser.add_argument("--num_train_epochs",',
    "def set_seed(",
    "def factorial",
]
for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors='pt').to(device).input_ids
    outputs = model.generate(
        input_ids=input_ids,
        max_length=64 + len(input_ids[0]),
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
        length_penalty=2.0,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Input :", prompt)
    print("Output:", decoded)
    print("=" * 20)
