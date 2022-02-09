[![PyPI version](https://badge.fury.io/py/code-autocomplete.svg)](https://badge.fury.io/py/code-autocomplete)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/code-autocomplete.svg)](https://github.com/shibing624/code-autocomplete/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_vesion](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/code-autocomplete.svg)](https://github.com/shibing624/code-autocomplete/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)

# Code AutoComplete
code-autocomplete, text to vector.

文本向量表征工具，把文本转化为向量矩阵，是文本进行计算机处理的第一步。

**code-autocomplete**实现了Word2Vec、RankBM25、BERT、Sentence-BERT、CoSENT等多种文本表征、文本相似度计算模型，并在文本语义匹配（相似度计算）任务上比较了各模型的效果。


**Guide**
- [Question](#Question)
- [Solution](#Solution)
- [Feature](#Feature)
- [Evaluate](#Evaluate)
- [Install](#install)
- [Usage](#usage)
- [Contact](#Contact)
- [Citation](#Citation)
- [Reference](#reference)

# Question
文本向量表示咋做？文本匹配任务用哪个模型效果好？

许多NLP任务的成功离不开训练优质有效的文本表示向量。特别是文本语义匹配（Semantic Textual Similarity，如paraphrase检测、QA的问题对匹配）、文本向量检索（Dense Text Retrieval）等任务。
# Solution


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

### 1. 计算文本向量

基于`pretrained model`计算文本向量


示例[computing_embeddings.py](./examples/computing_embeddings.py)



# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/code-autocomplete.svg)](https://github.com/shibing624/code-autocomplete/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：个人名称-公司-NLP* 进NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Citation

如果你在研究中使用了code-autocomplete，请按如下格式引用：

```latex
@software{code-autocomplete,
  author = {Xu Ming},
  title = {code-autocomplete: A Tool for Text to Vector},
  year = {2022},
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
- [将句子表示为向量（上）：无监督句子表示学习（sentence embedding）](https://www.cnblogs.com/llhthinker/p/10335164.html)
- [将句子表示为向量（下）：无监督句子表示学习（sentence embedding）](https://www.cnblogs.com/llhthinker/p/10341841.html)
- [A Simple but Tough-to-Beat Baseline for Sentence Embeddings[Sanjeev Arora and Yingyu Liang and Tengyu Ma, 2017]](https://openreview.net/forum?id=SyK00v5xx)
- [四种计算文本相似度的方法对比[Yves Peirsman]](https://zhuanlan.zhihu.com/p/37104535)
- [Improvements to BM25 and Language Models Examined](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf)
- [CoSENT：比Sentence-BERT更有效的句向量方案](https://kexue.fm/archives/8847)
- [谈谈文本匹配和多轮检索](https://zhuanlan.zhihu.com/p/111769969)
