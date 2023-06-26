# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys
import unittest

import torch

sys.path.append('..')
from codeassist.gpt2_coder import GPT2Coder


class IssueTestCase(unittest.TestCase):

    def test_code_predict(self):
        prompts = [
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
        ]
        infer = GPT2Coder("shibing624/code-autocomplete-gpt2-base")
        results = []
        for prompt in prompts:
            res = infer.generate(prompt)
            print("Query:", prompt)
            print("Result:", res[0])
            print("=" * 20)
            results.append(res[0])
        self.assertEqual(len(results), 3)


if __name__ == '__main__':
    unittest.main()
