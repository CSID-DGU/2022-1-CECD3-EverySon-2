from typing import List, Dict, Any
import torch
import torch.nn as nn
from omegaconf import omegaconf
import hydra
from fastapi import APIRouter
from pyrootutils import pyrootutils
from pydantic import BaseModel
import sys
from transformers import AutoTokenizer
from typing import List

sys.path.append("model/")

router = APIRouter(
    responses={404: {"description": "Not found"}},
)

class_names=[
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


def load_model():
    _model = None
    _tokenizer = None
    path = "model/configs"
    cfg_model = omegaconf.OmegaConf.load(f"{path}/CoMPM.yaml")
    _model = hydra.utils.instantiate(cfg_model)
    # _model = hydra.utils.instantiate(cfg_model, num_labels=len(class_names), class_names=class_names)
    # _model = _model.load_from_checkpoint(checkpoint_path = "checkpoint/model.ckpt")
    _tokenizer = AutoTokenizer.from_pretrained(cfg_model.net.pretrained_model_name_or_path)
    return _model, _tokenizer


model, tokenizer = load_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)


class Chat(BaseModel):
    speaker: List[int]
    text: List[str]


def _encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]
    ids = tokenizer.convert_tokens_to_ids(truncated)

    return [tokenizer.cls_token_id] + ids


def _padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)

    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]

        pad_ids.append(ids+add_ids)

    return torch.tensor(pad_ids)


def preprocess(speaker, text):
    context_speaker, context = speaker, text
    now_speaker = context_speaker[-1]
    speaker_utt_list = []
    inputString = ""
    for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
        inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
        inputString += utt + " "
        if turn<len(context_speaker)-1 and speaker == now_speaker:
            speaker_utt_list.append(_encode_right_truncated(utt, tokenizer, 512))

    concat_string = inputString.strip()

    data_dict = {}

    data_dict["input_token"] = torch.tensor(_encode_right_truncated(concat_string, tokenizer, 512))
    data_dict["speaker_token"] = _padding(speaker_utt_list, tokenizer)
    return data_dict["input_token"].unsqueeze(0), data_dict["speaker_token"].unsqueeze(0)


@router.on_event("startup")
async def startup_event():
    input_token, speaker_token = preprocess([0], ["안녕하세요, 반갑습니다."])
    model.forward(input_token.cuda(), speaker_token.cuda())
    model.forward(input_token.cuda(), speaker_token.cuda())

@router.post("/")
async def predict_label(chat: Chat):
    assert len(chat.speaker) == len(chat.text)
    input_token, speaker_token = preprocess(chat.speaker, chat.text)
    outputs = model.forward(input_token.cuda(), speaker_token.cuda())
    y = outputs.detach().cpu().squeeze().argmax(dim=-1)
    label = class_names[y]
    return {"label": label}
