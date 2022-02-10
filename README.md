[![PyPI version](https://badge.fury.io/py/code-autocomplete.svg)](https://badge.fury.io/py/code-autocomplete)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/code-autocomplete.svg)](https://github.com/shibing624/code-autocomplete/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_vesion](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/code-autocomplete.svg)](https://github.com/shibing624/code-autocomplete/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

# Code AutoComplete
code-autocomplete, a code completion plugin for Python.

**code-autocomplete**实现了Python代码行粒度和块粒度自动补全功能。


**Guide**
- [Feature](#Feature)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [Citation](#Citation)
- [Reference](#reference)

# Feature


# Demo

http://42.193.145.218/product/short_text_sim/

# Install
```
pip3 install -U code-autocomplete
```

or

```
git clone https://github.com/shibing624/code-autocomplete.git
cd code-autocomplete
python3 setup.py install
```


# Usage

### Code Completion

基于`GPT-2 model`预测整行代码


示例[gpt2_demo.py](./examples/gpt2_demo.py)
```python
import sys

sys.path.append('..')
from autocomplete.gpt2 import predict

if __name__ == '__main__':
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
    for prompt in prompts:
        res = predict(prompt, model_dir='shibing624/code-autocomplete-gpt2')
        print("\n\n======================\n\n")
        print("Query:", prompt)
        print("\nResult:, res")
        print("\n\n======================\n\n")

```


# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/code-autocomplete.svg)](https://github.com/shibing624/code-autocomplete/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：个人名称-公司-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Citation

如果你在研究中使用了code-autocomplete，请按如下格式引用：

```latex
@misc{code-autocomplete,
  author = {Xu Ming},
  title = {code-autocomplete: Code AutoComplete with GPT model},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/shibing624/code-autocomplete},
}
```

# License


授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加code-autocomplete的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在`tests`添加相应的单元测试
 - 使用`python setup.py test`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

# Reference
- [https://github.com/galois-autocompleter/galois-autocompleter](https://github.com/galois-autocompleter/galois-autocompleter)
