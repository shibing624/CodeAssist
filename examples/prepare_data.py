# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import numpy as np
import sys

sys.path.append("..")
from autocomplete import create_dataset


def main(limit_size=10):
    create_dataset.create_folders()
    try:
        create_dataset.progressive(limit_size=limit_size)
    except KeyboardInterrupt:
        pass
    source_files = create_dataset.get_python_files()
    print(f'Source_files size: {len(source_files)}')
    np.random.shuffle(source_files)
    train_valid_split = int(len(source_files) * 0.9)
    train_file = 'download/train.txt'
    valid_file = 'download/valid.txt'
    create_dataset.concat_and_save(train_file, source_files[:train_valid_split])
    create_dataset.concat_and_save(valid_file, source_files[train_valid_split:])
    print(f'Save train file: {train_file}, valid file: {valid_file}')


if __name__ == '__main__':
    main(limit_size=10)
