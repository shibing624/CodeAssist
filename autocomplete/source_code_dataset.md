---
annotations_creators:
- no-annotation
language_creators:
- crowdsourced
languages:
- en
licenses:
- cc-by-4-0
- gfdl-1-3-or-later
multilinguality:
- monolingual
size_categories:
- 100M<n<200M
source_datasets:
- https://github.com/shibing624/code-autocomplete
- https://github.com/bharathgs/Awesome-pytorch-list
- https://github.com/akullpp/awesome-java
- https://github.com/fffaraz/awesome-cpp
task_categories:
- sequence-modeling
task_ids:
- language-modeling
---
# Dataset Card for "SourceCode"
## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description
- **Repository:** [code-autocomplete](https://github.com/shibing624/code-autocomplete)
- **Leaderboard:** [leaderboard](https://github.com/shibing624/code-autocomplete) (located on the homepage)
- **Size of downloaded dataset files:** 105 MB
- **Total amount of disk used:** 570 MB

### Dataset Summary

Source code dataset is a collection of Github awesome repos, it contains Python, Java, C++, and other programming languages.
This dataset can be used in different NLP tasks like language modeling and text generation tasks.

data source:

- PYTHON_CODE: https://github.com/bharathgs/Awesome-pytorch-list
- JAVA_CODE: https://github.com/akullpp/awesome-java
- CPP_CODE: https://github.com/fffaraz/awesome-cpp


### Supported Tasks and Leaderboards
- language modeling 
- code generation tasks, **Leaderboard:** [code-autocomplete](https://github.com/shibing624/code-autocomplete)

### Languages

- programming languages: Python, Java, C++
- natural language: English

## Dataset Structure
### Data Instances
An example of 'train' looks as follows.
```
This example was too long and was cropped:

{
    "text": """
import json
import argparse


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--model-file',
        required=True,
        help=(
            'A pt file from '
            'https://github.com/pytorch/fairseq/tree/main/examples/hubert'
        )
    )
    return parser.parse_args()
    """
}
```
### Data Fields
The data fields are the same among all splits.
- `text`: a `string` feature.
### Data Splits
#### python
```shell
$ wc -l python/*
   10000 python/test.txt
 5215412 python/train.txt
   10000 python/valid.txt
 5235412 total
```
#### java
```shell
$ wc -l java/*  
  950083 java/test.txt
 2802880 java/train.txt
  940803 java/valid.txt
 4693766 total
```
#### cpp
```shell
$ wc -l cpp/* 
 1060014 cpp/test.txt
 3119241 cpp/train.txt
 1099124 cpp/valid.txt
 5278379 total
```
## Dataset Creation
### Curation Rationale
As code generation dataset, I upload it to huggingface datasets.
### Source Data
#### Initial Data Collection and Normalization
#### Who are the source language producers?
Citation:

APA:
```latex
Xu, M. code-autocomplete: Code AutoComplete with GPT2 model (Version 0.0.4) [Computer software]. https://github.com/shibing624/code-autocomplete
```

BibTeX:
```latex
@software{Xu_code-autocomplete_Code_AutoComplete,
author = {Xu, Ming},
title = {code-autocomplete: Code AutoComplete with GPT2 model},
url = {https://github.com/shibing624/code-autocomplete},
version = {0.0.4}
}
```

### Annotations
#### Annotation process
#### Who are the annotators?
nobody
### Personal and Sensitive Information
## Considerations for Using the Data
### Social Impact of Dataset
This dataset was developed as a benchmark for evaluating code generation model.
### Discussion of Biases
### Other Known Limitations
## Additional Information
### Dataset Curators

Github awesome programing code repos.

### Licensing Information

For research use only.

### Contributions
Thanks to [@shibing624](https://github.com/shibing624) add this dataset.