from fastapi import FastAPI
app = FastAPI()
from fastapi import FastAPI
from pydantic import BaseModel
from run_generation2 import generate_text, load_model

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn

import numpy as np
import torch
np.random.seed(42)
torch.manual_seed(42)
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device='cuda'

#tok = GPT2Tokenizer.from_pretrained("content/weights_push_v_may")
#model = GPT2LMHeadModel.from_pretrained("content/weights_push_v_may").to(device)
model, tokenizer = load_model(no_cuda=False)

def push(text):
  '''
  repetition_penalty = 2.6
  temperature = temp
  top_k =4 

  inpt = tok.encode(text, return_tensors="pt").to(device)

  max_length=lenght

  out = model.generate(inpt,max_length= max_length, 
                       repetition_penalty=repetition_penalty, 
                       do_sample=True, top_k=top_k, top_p=0.95, 
                       temperature=temperature,num_return_sequences=num)
  decoded = tok.decode(out[0])
  return decoded
  '''
  
  return  generate_text(
            model,
            tokenizer,
            model_type='gpt2',
            length=20,
            prompt=text,
            temperature=0.9,
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
    return "go to /push/?message=пидр"


class Req(BaseModel):
  text: str
  userId: int

@app.post('/generate')
async def detect_spam_query(req: Req):
  #jsonify({'result': result})
  return {"result":push(req.text)}

import nest_asyncio
#from pyngrok import ngrok

nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=5000,timeout_keep_alive=10000)

