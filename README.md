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
- GPT2-based code completion
- Code completion for Python, other language is coming soon
- Line and block completion
- Train(Fine-tune GPT2) and predict model with your own data

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


基于GPT2模型预测补全代码，通过如下命令调用：

```python
from autocomplete.gpt2 import Infer
m = Infer(model_name="gpt2", model_dir="shibing624/code-autocomplete-gpt2-base", use_cuda=False)
i = m.predict('import torch.nn as')
print(i)
```

output:
```shell
import torch.nn as nn
```
当然，你也可使用官方的huggingface/transformers调用：

*Please use 'GPT2' related functions to load this model!*

```python
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("shibing624/code-autocomplete-gpt2-base")
model = GPT2LMHeadModel.from_pretrained("shibing624/code-autocomplete-gpt2-base")
model.to(device)
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
    "def factorial(n):",
]
for prompt in prompts:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    outputs = model.generate(input_ids=input_ids,
                             max_length=64 + len(prompt),
                             temperature=1.0,
                             top_k=50,
                             top_p=0.95,
                             repetition_penalty=1.0,
                             do_sample=True,
                             num_return_sequences=1,
                             length_penalty=2.0,
                             early_stopping=True)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded)
    print("=" * 20)
```

output:
```python
from torch import nn
class LSTM(Module):
    def __init__(self, *,
                 n_tokens: int,
                 embedding_size: int,
                 hidden_size: int,
                 n_layers: int):
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

====================

import numpy as np
import torch
import torch.nn as nn

====================
...
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
