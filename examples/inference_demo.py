# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

sys.path.append('..')
from codeassist import GPT2Coder, WizardCoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="wizard", help='wizard or gpt2')
    parser.add_argument('--model_name', type=str, default="WizardLM/WizardCoder-15B-V1.0", help='model name or path')
    parser.add_argument('--max_length', type=int, default=128, help='maximum length for code generation')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for sampling-based code generation')
    args = parser.parse_args()
    print(args)
    if args.model_type == 'wizard':
        model = WizardCoder(args.model_name)
    else:
        model = GPT2Coder(args.model_name)

    # generate code
    while True:
        print(f'Enter the context code (exit or python code)')
        context = input(">>> ")
        if context == "exit":
            break
        generated_codes = model.generate(context, temperature=args.temperature, max_length=args.max_length)
        print("Generated code:")
        for i, code in enumerate(generated_codes):
            print("{}:\n {}".format(i + 1, code))
        print("=" * 20)
