# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

sys.path.append('..')
from autocomplete.gpt2_coder import GPT2Coder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_dir', type=str, default="outputs-fine-tuned", help='the path to load fine-tuned model')
    parser.add_argument('--max_length', type=int, default=64, help='maximum length for code generation')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for sampling-based code generation')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='number of sequences to be generated')
    args = parser.parse_args()

    # load fine-tuned model and tokenizer from path specified by --model_dir
    model = GPT2Coder(args.model_dir, args.max_length)

    # generate code
    while True:
        print(f'Enter the context code (exit or python code)')
        context = input(">>> ")
        if context == "exit":
            break
        generated_codes = model.generate(context, temperature=args.temperature,
                                         num_return_sequences=args.num_return_sequences)
        print("Generated code:")
        for i, code in enumerate(generated_codes):
            print("{}:\n {}".format(i + 1, code))
        print("=" * 20)
