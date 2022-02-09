# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import sys
import unittest
from time import time

sys.path.append('..')
from autocomplete import Word2Vec, SBert
from autocomplete.cosent.data_helper import load_test_data

pwd_path = os.path.abspath(os.path.dirname(__file__))
sts_test_path = os.path.join(pwd_path, '../code-autocomplete/data/STS-B/STS-B.test.data')


class QPSEncoderTestCase(unittest.TestCase):
    def test_cosent_speed(self):
        """测试cosent_speed"""
        sents1, sents2, labels = load_test_data(sts_test_path)
        m = SBert('shibing624/code-autocomplete-base-chinese')
        sents = sents1 + sents2
        print('sente size:', len(sents))
        t1 = time()
        m.encode(sents)
        spend_time = time() - t1
        print('spend time:', spend_time, ' seconds')
        print('cosent_sbert qps:', len(sents) / spend_time)



if __name__ == '__main__':
    unittest.main()
