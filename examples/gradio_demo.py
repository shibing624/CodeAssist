"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import gradio as gr
from autocomplete.gpt2_coder import GPT2Coder

model = GPT2Coder("shibing624/code-autocomplete-gpt2-base")


def ai_text(text):
    gen_text = model.generate(text)[0]
    print(text, ' => ', gen_text)
    return gen_text


if __name__ == '__main__':
    print(ai_text('import torch.nn as'))

    examples = [
        ['import torch.nn as'],
        ['parser.add_argument("--num_train_epochs",'],
        ['torch.device('],
        ['def set_seed('],
    ]

    output_text = gr.outputs.Textbox()
    gr.Interface(ai_text, "textbox", output_text,
                 # theme="grass",
                 title="Code Autocomplete Model shibing624/code-autocomplete-gpt2-base",
                 description="Copy or input python code here. Submit and the machine will generate code.",
                 article="Link to <a href='https://github.com/shibing624/code-autocomplete' style='color:blue;' target='_blank\'>Github REPO</a>",
                 examples=examples).launch()