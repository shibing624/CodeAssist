"""
@author:XuMing(xuming624@qq.com)
@description: Server
"""
import argparse
import uvicorn
import sys
import os
from fastapi import FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
import torch
from loguru import logger

sys.path.append('..')
from autocomplete.gpt2_coder import GPT2Coder

pwd_path = os.path.abspath(os.path.dirname(__file__))
use_cuda = torch.cuda.is_available()
# Use finetuned GPT2 model
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="shibing624/code-autocomplete-gpt2-base",
                    help="Model save dir or model name")
args = parser.parse_args()
model = GPT2Coder(args.model_name_or_path)

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
        generated = model.generate(q)
        result_dict = generated[0]
        logger.debug(f"Successfully autocomplete, q:{q}, res:{result_dict}")
        return result_dict
    except Exception as e:
        logger.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8001)
