# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import sys

sys.path.append("..")
from autocomplete import create_dataset
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="download", help="Save dataset directory")
    parser.add_argument("--num_repos", type=int, default=3, help="Number of repos to use")
    parser.add_argument("--code", default="python", const='python', nargs='?',
                        choices=['python', 'java', 'cpp'], help="Download code language source code dataset")
    args = parser.parse_args()
    print(args)

    try:
        sources = create_dataset.get_source_code_by_language(code_languages=args.code,
                                                             save_dir=args.save_dir,
                                                             each_limit_repos=args.num_repos
                                                             )
    except KeyboardInterrupt:
        sources = dict()
        pass
    X = sources[f"{args.code}"]
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
    train_file = f'{args.save_dir}/{args.code}/train.txt'
    valid_file = f'{args.save_dir}/{args.code}/valid.txt'
    test_file = f'{args.save_dir}/{args.code}/test.txt'
    create_dataset.merge_and_save(X_train, train_file)
    create_dataset.merge_and_save(X_val, valid_file)
    create_dataset.merge_and_save(X_test, test_file)
    print(f'Save train file: {train_file}, valid file: {valid_file}, test file: {test_file}')


if __name__ == '__main__':
    main()
