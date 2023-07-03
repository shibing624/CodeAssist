[**🇨🇳中文**](https://github.com/shibing624/codeassist/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/codeassist/blob/main/README_EN.md) | [**📖文档/Docs**](https://github.com/shibing624/codeassist/wiki) | [**🤖模型/Models**](https://huggingface.co/shibing624) 

<div align="center">
  <a href="https://github.com/shibing624/codeassist">
    <img src="https://github.com/shibing624/codeassist/blob/main/docs/codeassist.png" height="130" alt="Logo">
  </a>
</div>

-----------------

# CodeAssist: Advanced Code Completion Tool
[![PyPI version](https://badge.fury.io/py/codeassist.svg)](https://badge.fury.io/py/codeassist)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/codeassist.svg)](https://github.com/shibing624/codeassist/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_vesion](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/codeassist.svg)](https://github.com/shibing624/codeassist/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

## Introduction

**CodeAssist** is an advanced code completion tool that intelligently provides high-quality code completions for Python, Java, and C++ and so on. 

CodeAssist 是一个高级代码补全工具，高质量为 Python、Java 和 C++ 等编程语言补全代码


## Features

- GPT based code completion
- Code completion for `Python`, `Java`, `C++`, `javascript` and so on
- Line and block code completion
- Train(Fine-tuning) and predict model with your own data

### Release Models

| Arch   | BaseModel         | Model                                                                                                                   | Model Size | 
|:-------|:------------------|:------------------------------------------------------------------------------------------------------------------------|:----------:|
| GPT   | gpt2              | [shibing624/code-autocomplete-gpt2-base](https://huggingface.co/shibing624/code-autocomplete-gpt2-base)                 |   487MB    |
| GPT   | distilgpt2        | [shibing624/code-autocomplete-distilgpt2-python](https://huggingface.co/shibing624/code-autocomplete-distilgpt2-python) |   319MB    |
| GPT   | bigcode/starcoder | [WizardLM/WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)                                   |    29GB    |


### Demo

HuggingFace Demo: https://huggingface.co/spaces/shibing624/code-autocomplete

backend model: `shibing624/code-autocomplete-gpt2-base`

## Install

```shell
pip install torch # conda install pytorch
pip install -U codeassist
```

or

```shell
git clone https://github.com/shibing624/codeassist.git
cd CodeAssist
python setup.py install
```

## Usage

### WizardCoder model

WizardCoder-15b is fine-tuned `bigcode/starcoder` with alpaca code data, you can use the following code to generate code:

example: [examples/wizardcoder_demo.py](https://github.com/shibing624/CodeAssist/blob/main/examples/wizardcoder_demo.py)

```python
import sys

sys.path.append('..')
from codeassist import WizardCoder

m = WizardCoder("WizardLM/WizardCoder-15B-V1.0")
print(m.generate('def load_csv_file(file_path):')[0])
```

output:


```python
import csv

def load_csv_file(file_path):
    """
    Load data from a CSV file and return a list of dictionaries.
    """
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)
        # Initialize an empty list to store the data
        data = []
        # Iterate over each row of data
        for row in csv_reader:
            # Append the row of data to the list
            data.append(row)
    # Return the list of data
    return data
```

model output is impressively effective, it currently supports English and Chinese input, you can enter instructions or code prefixes as required.

### distilgpt2 model


distilgpt2 fine-tuned code autocomplete model, you can use the following code:

example: [examples/distilgpt2_demo.py](https://github.com/shibing624/CodeAssist/blob/main/examples/distilgpt2_demo.py)

```python
import sys

sys.path.append('..')
from codeassist import GPT2Coder

m = GPT2Coder("shibing624/code-autocomplete-distilgpt2-python")
print(m.generate('import torch.nn as')[0])
```

output:

```shell
import torch.nn as nn
import torch.nn.functional as F
```

### Use with huggingface/transformers：

example: [examples/use_transformers_gpt2.py](https://github.com/shibing624/CodeAssist/blob/main/examples/use_transformers_gpt2.py)

### Train Model
#### Train WizardCoder model
example: [examples/training_wizardcoder_mydata.py](https://github.com/shibing624/CodeAssist/blob/main/examples/training_wizardcoder_mydata.py)

```shell
cd examples
CUDA_VISIBLE_DEVICES=0,1 python training_wizardcoder_mydata.py --do_train --do_predict --num_epochs 1 --output_dir outputs-wizard --model_name WizardLM/WizardCoder-15B-V1.0
```

- GPU memory: 31GB
- finetune need 2*V100(32GB)
- inference need 1*V100(32GB)

#### Train distilgpt2 model
example: [examples/training_gpt2_mydata.py](https://github.com/shibing624/CodeAssist/blob/main/examples/training_gpt2_mydata.py)

```shell
cd examples
python training_gpt2_mydata.py --do_train --do_predict --num_epochs 15 --output_dir outputs-gpt2 --model_name gpt2
```

PS: fine-tuned result model is GPT2-python: [shibing624/code-autocomplete-gpt2-base](https://huggingface.co/shibing624/code-autocomplete-gpt2-base), 
I spent about 24 hours with V100 to fine-tune it. 


### Server

start FastAPI server:

example: [examples/server.py](https://github.com/shibing624/CodeAssist/blob/main/examples/server.py)

```shell
cd examples
python server.py
```

open url: http://0.0.0.0:8001/docs

![api](https://github.com/shibing624/CodeAssist/blob/main/docs/api.png)



## Dataset

This allows to customize dataset building. Below is an example of the building process.

Let's use Python codes from [Awesome-pytorch-list](https://github.com/bharathgs/Awesome-pytorch-list)

1. We want the model to help auto-complete codes at a general level. The codes of The Algorithms suits the need.
2. This code from this project is well written (high-quality codes).

dataset tree:

```shell
examples/download/python
├── train.txt
└── valid.txt
└── test.txt
```

There are three ways to build dataset:
1. Use the huggingface/datasets library load the dataset
huggingface datasets [https://huggingface.co/datasets/shibing624/source_code](https://huggingface.co/datasets/shibing624/source_code)

```python
from datasets import load_dataset
dataset = load_dataset("shibing624/source_code", "python") # python or java or cpp
print(dataset)
print(dataset['test'][0:10])
```

output:
```shell
DatasetDict({
    train: Dataset({
        features: ['text'],
        num_rows: 5215412
    })
    validation: Dataset({
        features: ['text'],
        num_rows: 10000
    })
    test: Dataset({
        features: ['text'],
        num_rows: 10000
    })
})
{'text': [
"            {'max_epochs': [1, 2]},\n", 
'            refit=False,\n', '            cv=3,\n', 
"            scoring='roc_auc',\n", '        )\n', 
'        search.fit(*data)\n', 
'', 
'    def test_module_output_not_1d(self, net_cls, data):\n', 
'        from skorch.toy import make_classifier\n', 
'        module = make_classifier(\n'
]}
```

2. Download dataset from Cloud

| Name | Source | Download | Size |
| :------- | :--------- | :---------: | :---------: |
| Python+Java+CPP source code | Awesome-pytorch-list(5.22 Million lines) | [github_source_code.zip](https://github.com/shibing624/codeassist/releases/download/0.0.4/source_code.zip) | 105M |

download dataset and unzip it, put to `examples/`.

3. Get source code from scratch and build dataset

[prepare_code_data.py](https://github.com/shibing624/CodeAssist/blob/main/examples/prepare_code_data.py)

```shell
cd examples
python prepare_code_data.py --num_repos 260
```


## Contact

- Issue(建议)
  ：[![GitHub issues](https://img.shields.io/github/issues/shibing624/codeassist.svg)](https://github.com/shibing624/codeassist/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我： 加我*微信号：xuming624, 备注：个人名称-公司-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />

## Citation

如果你在研究中使用了codeassist，请按如下格式引用：

APA:
```latex
Xu, M. codeassist: Code AutoComplete with GPT model (Version 1.0.0) [Computer software]. https://github.com/shibing624/codeassist
```

BibTeX:
```latex
@software{Xu_codeassist,
author = {Ming Xu},
title = {CodeAssist: Code AutoComplete with Generation model},
url = {https://github.com/shibing624/codeassist},
version = {1.0.0}
}
```

## License
This repository is licensed under the [The Apache License 2.0](LICENSE).

Please follow the [Attribution-NonCommercial 4.0 International](https://github.com/nlpxucan/WizardLM/blob/main/WizardCoder/MODEL_WEIGHTS_LICENSE) to use the WizardCoder model.


## Contribute

项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

- 在`tests`添加相应的单元测试
- 使用`python setup.py test`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

## Reference

- [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple)
- [galois-autocompleter](https://github.com/galois-autocompleter/galois-autocompleter)
- [WizardLM/WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0)
