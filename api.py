#!/usr/local/bin/python3
# coding: utf8
from fastapi import FastAPI
app = FastAPI()
from fastapi import FastAPI
import nest_asyncio
from pydantic import BaseModel
from run_generation import generate_text, load_model

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn

import numpy as np
#import torch
#np.random.seed(42)
#torch.manual_seed(42)
#from transformers import GPT2LMHeadModel, GPT2Tokenizer

device='cuda'

#tok = GPT2Tokenizer.from_pretrained("content/weights_push_v_may")
#model = GPT2LMHeadModel.from_pretrained("content/weights_push_v_may").to(device)
model, tokenizer = load_model(no_cuda=False)

def push(text: str, length: int, temperature: float):
  """function to generate new text from input with hyperparams

  Args:
      text (str): input text
      length (int): len of generation
      temperature (float): "creativness" of generation
  Returns:
      str: generated text
  """  
  return  generate_text(
            model,
            tokenizer,
            model_type='gpt2',
            length=length,
            prompt=text,
            temperature=temperature,
            no_cuda=False
            )


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def main():
    return "This is simple API to interact ponyfiction GPT-2. Send generation requests on /generate"


class Req(BaseModel):
  text: str
  lenght: int
  temperature: float

@app.post('/generate')
async def generate(req: Req):
  return {"result":push(req.text, req.lenght, req.temperature)}



nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=5000,timeout_keep_alive=10000)

