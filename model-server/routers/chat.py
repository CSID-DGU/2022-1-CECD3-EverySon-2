from typing import List, Dict, Any
import torch
import torch.nn as nn
from omegaconf import omegaconf
import hydra
from fastapi import APIRouter
from pydantic import BaseModel
import sys
from transformers import AutoTokenizer
from typing import List
from chatbot import Chatbot
import hydra
import omegaconf
import sys

sys.path.append("model/")

cfg = omegaconf.OmegaConf.load("model/configs/configs.yaml")
chatbot = Chatbot(cfg)

router = APIRouter(
    responses={404: {"description": "Not found"}},
)


class Chat(BaseModel):
    speaker: List[int]
    text: List[str]


@router.on_event("startup")
async def startup_event():
    chat = Chat(speaker = [0, 1], text=["오늘 기분은 어때?", "너무너무 행복한 하루였어!"])
    chatbot.emotion_recognition(chat)
    
@router.post("/")
async def predict_label(chat: Chat):
    assert len(chat.speaker) == len(chat.text)
    label = chatbot.emotion_recognition(chat)
    return {"label": label}
