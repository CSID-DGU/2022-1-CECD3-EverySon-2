from pickletools import optimize
from turtle import forward
import numpy as np
import pandas as pd
import sentence_transformers as st
from transformers import AutoTokenizer, AutoModel
import urllib.request
from tqdm import tqdm
from typing import Any
import torch
import pytorch_lightning as pl


class CosSimChatbot(pl.LightningModule):
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 embedding_path: str,
                 *args, **kwargs,
                 ) -> None:
        super().__init__(*args, **kwargs)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        chatbot_embedding = torch.load(embedding_path)
        self.register_buffer("chatbot_embedding", chatbot_embedding)

    def step(self, encoded_input):
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def predict_step(self, batch: Any, batch_idx) -> Any:
        sentence_embeddings = self.step(batch)
        score = st.util.cos_sim(sentence_embeddings, self.chatbot_embedding)
        best_score_idx = torch.argmax(score)
        return best_score_idx

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        
if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    cfg = omegaconf.OmegaConf.create({"paths": {"data_dir": "null"},
                                      "models": "null"})
    cfg.paths.data_dir = "/home/sj/Project/2022-1-CECD3-EverySon-2/model/data/"
    cfg.model = omegaconf.OmegaConf.load(root / "configs" / "model" / "cos_sim_chatbot.yaml")
    model = hydra.utils.instantiate(cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path)
    
    text = "아 기분이 안좋네..."
    label = "슬픔"

    tokenizer_input = text + " [SEP] " + label
    tokenized_sentences = tokenizer(
        tokenizer_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=256,)

    y = model.predict_step(tokenized_sentences, batch_idx=0)
    print(y)