# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import sys

sys.path.append('..')
from codeassist import WizardCoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="WizardLM/WizardCoder-15B-V1.0",
                        help="Model arch, gpt2, gpt2-medium, distilgpt2 or WizardLM/WizardCoder-15B-V1.0")
    parser.add_argument("--train_file", type=str, default="data/code_alpaca_20k_50.jsonl", help="Train file path")
    parser.add_argument("--valid_file", type=str, default="data/code_alpaca_20k_50.jsonl", help="Valid file path")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predict.")
    parser.add_argument("--output_dir", type=str, default="./outputs-finetuned-wizardcoder/", help="output dir")
    parser.add_argument("--num_epochs", type=int, default=5, help="Num of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    args = parser.parse_args()
    print(args)

    if args.do_train:
        model = WizardCoder(model_name_or_path=args.model_name)
        model.train_model(
            args.train_file,
            args.output_dir,
            eval_file=args.valid_file,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size
        )
        print(f"model saved to {args.output_dir}")
    if args.do_predict:
        model = WizardCoder(model_name_or_path=args.model_name, peft_name=args.output_dir)
        prompts = [
            "def load_csv_file(file_path):",
            "write a C++ code to sum 1 to 12.",
            "写个python的快排算法",
            "生成4到400之间的随机数，用java和python写代码",
        ]
        for prompt in prompts:
            outputs = model.generate(prompt)
            print("Input :", prompt)
            print("Output:", outputs[0])
            print("=" * 20)
