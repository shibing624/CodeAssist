# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import argparse
import os
import sys
import unittest
from time import time

sys.path.append('..')
from autocomplete.gpt2_coder import GPT2Coder

pwd_path = os.path.abspath(os.path.dirname(__file__))

test_file = os.path.join(pwd_path, 'test.txt')

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="shibing624/code-autocomplete-gpt2-base",
                    help="Model directory to export parameters from.")
args = parser.parse_args()
print(args)

model = GPT2Coder(args.model_name)


def load_data(file_path):
    res = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            res.append(line)
    return res


class ModelTestCase(unittest.TestCase):
    def test_model_infer(self):
        """test_model_infer"""
        codes = load_data(test_file)
        codes = codes[100:110]
        t1 = time()
        for prompt in codes:
            res = model.generate(prompt)
            print("Query:", prompt)
            print("Result:", res[0])
            print("=" * 20)
        spend_time = time() - t1
        print('spend time:', spend_time, ' seconds')
        print('size:', len(codes), ' qps:', len(codes) / spend_time)


if __name__ == '__main__':
    unittest.main()
