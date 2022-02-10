"""
@author:XuMing(xuming624@qq.com)
@description: Server
"""
import uvicorn
import sys
import os
from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
import torch
from simpletransformers.language_generation import LanguageGenerationModel
from simpletransformers.language_modeling import LanguageModelingModel
from loguru import logger

use_cuda = torch.cuda.is_available()
# Use finetuned model
gpt2 = LanguageGenerationModel("gpt2", "outputs/fine-tuned", args={"max_length": 200}, use_cuda=use_cuda)

# define the app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


@app.get('/')
async def index():
    return {"message": "index, docs url: /docs"}


@app.get('/autocomplete')
async def autocomplete(q: str = Query(..., min_length=1, max_length=512, title='query')):
    try:
        # Generate text using the model. Verbose set to False to prevent logging generated sequences.
        generated = gpt2.generate(q, verbose=False)
        generated = ".".join(generated[0].split(".")[:-1]) + "."
        print(generated)
        result_dict = generated
        logger.debug(f"Successfully autocomplete, q:{q}, res:{result_dict}")
        return result_dict
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000)
