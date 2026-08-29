import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure src is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.mt5_translate import load_mt5_model, mt5_translate
from src.data.hcmus_translate import hcmus_translate

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    load_mt5_model()
    yield
    # Clean up on shutdown if necessary

app = FastAPI(title="Sino-Nom to Vietnamese API", lifespan=lifespan)

# Allow CORS for the Vite dev server (and potentially other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str
    model: str = "mt5" # default model

class HCMUSRequest(BaseModel):
    text: str

@app.post("/api/translate/local")
async def translate_local(req: TranslationRequest):
    if not req.text.strip():
        return {"result": ""}
    
    if req.model == "mt5":
        res = mt5_translate(req.text)
        return {"result": res}
    else:
        return {"result": "Model not supported."}

@app.post("/api/translate/hcmus")
async def translate_hcmus(req: HCMUSRequest):
    if not req.text.strip():
        return {"result": ""}
    
    res = hcmus_translate(req.text)
    return {"result": res}
