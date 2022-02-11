# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest
from time import time
import torch

sys.path.append('..')
from autocomplete.gpt2 import Infer

pwd_path = os.path.abspath(os.path.dirname(__file__))
use_cuda = torch.cuda.is_available()

test_file = os.path.join(pwd_path, '../examples/test.txt')
model = Infer(model_name="gpt2", model_dir="shibing624/code-autocomplete-gpt2-base", use_cuda=use_cuda)


def load_data(file_path):
    res = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            res.append(line)
    return res


class QPSTestCase(unittest.TestCase):
    def test_code_infer_speed(self):
        """Test code_infer_speed"""
        codes = load_data(test_file)
        codes = codes[100:110]
        t1 = time()
        for prompt in codes:
            res = model.predict(prompt)
            print("Query:", prompt)
            print("Result:", res)
            print("=" * 20)
        spend_time = time() - t1
        print('spend time:', spend_time, ' seconds')
        print('size:', len(codes), ' qps:', len(codes) / spend_time)


if __name__ == '__main__':
    unittest.main()
