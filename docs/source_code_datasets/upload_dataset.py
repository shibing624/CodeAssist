# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

"""Code AutoComplete Python dataset Corpus.(code_autocomplete)"""

import os

import datasets

_DESCRIPTION = """纯文本数据，内容：高质量编程源代码，包括Python，Java，CPP源代码"""

PYTHON_HOME = "https://github.com/bharathgs/Awesome-pytorch-list"
JAVA_HOME = "https://github.com/akullpp/awesome-java"
CPP_HOME = "https://github.com/fffaraz/awesome-cpp"

_CITATION = "https://github.com/shibing624/code-autocomplete"

_DATA_URL = "https://github.com/shibing624/code-autocomplete/releases/download/0.0.4/source_code.zip"


class SourceCodeConfig(datasets.BuilderConfig):
    """BuilderConfig for NLI_zh"""

    def __init__(self, features, data_url, citation, url, **kwargs):
        """BuilderConfig for NLI_zh
        Args:
          features: `list[string]`, list of the features that will appear in the
            feature dict. Should not include "label".
          data_url: `string`, url to download the zip file from.
          citation: `string`, citation for the data set.
          url: `string`, url for information about the data set.
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.features = features
        self.data_url = data_url
        self.citation = citation
        self.url = url


class SourceCode(datasets.GeneratorBasedBuilder):
    """The Natural Language Inference Chinese(NLI_zh) Corpus."""

    BUILDER_CONFIGS = [
        SourceCodeConfig(
            name="python",
            description=_DESCRIPTION,
            features=["text"],
            data_url=_DATA_URL,
            citation=_CITATION,
            url=PYTHON_HOME,
        ),
        SourceCodeConfig(
            name="java",
            description=_DESCRIPTION,
            features=["text"],
            data_url=_DATA_URL,
            citation=_CITATION,
            url=JAVA_HOME,
        ),
        SourceCodeConfig(
            name="cpp",
            description=_DESCRIPTION,
            features=["text"],
            data_url=_DATA_URL,
            citation=_CITATION,
            url=CPP_HOME,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                }
            ),
            homepage=self.config.url,
            citation=self.config.citation,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(self.config.data_url) or ""
        dl_dir = os.path.join(dl_dir, f"source_code/{self.config.name}")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, f"train.txt"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, f"valid.txt"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(dl_dir, f"test.txt"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, 'r', encoding="utf-8") as f:
            for idx, row in enumerate(f):
                if row.strip():
                    yield idx, {"text": row}
                else:
                    yield idx, {"text": ""}
