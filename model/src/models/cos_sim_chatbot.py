from os import device_encoding
from pickletools import optimize
from tkinter import Variable
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
from pathlib import Path


class CosSimChatbot(pl.LightningModule):
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 optimizer: torch.optim.Optimizer,
                 chatbot_embedding_path: str=None,
                 chatbot_embedding_out: str=None,
                 *args, **kwargs,
                 ) -> None:
        """CosSimChatbot init

        Args:
            pretrained_model_name_or_path (str): pretrained model
            optimizer (torch.optim.Optimizer): dummy optimizer
            chatbot_embedding_path (str, optional): parquet filepath(not dirpath).
                                                    Defaults to None(for training from scrach).
            chatbot_embedding_out (str, optional): parquet filepath(not dirpath).
                                                   Defaults to None(only requires when training from scrach).
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self._freeze_model()
        # load chatbot datas from csv file
        # - embedding   : embedding
        # - answer      : chatbot response
        if chatbot_embedding_path is not None:  # train from file && predict
            self.load_chatbot_data(chatbot_embedding_path)
        else:   # train from scrach
            chatbot_embeddings = torch.zeros(0)
            self.answer_dict = {}
            self.key = 0
            self.chatbot_embedding_out = chatbot_embedding_out
        # self.register_buffer("chatbot_embeddings", chatbot_embeddings)
            self.chatbot_embeddings = chatbot_embeddings
        self.embedding_list = []
        self.dummy_layer = torch.nn.Linear(1, 1, bias=False)

    def load_chatbot_data(self, chatbot_embedding_path):
        chatbot_df = self._read_chatbot_db(chatbot_embedding_path)
        self.chatbot_embeddings = torch.tensor(chatbot_df["embedding"])
        self.answer_dict = chatbot_df["answer"].to_dict()
        self.key = len(chatbot_df["answer"])
        self.chatbot_embedding_out = chatbot_embedding_path

    def _freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x) -> Any:
        return self.dummy_layer(x)

    def step(self, encoded_input):
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

    def training_step(self, batch, batch_idx):
        answer = batch.pop("answer")
        for ans in answer:
            self.answer_dict[self.key] = ans
            self.key += 1
        sentence_embeddings = self.step(batch)
        self.embedding_list.append(sentence_embeddings)
        return {"loss": self.forward(torch.zeros(1, device=self.device))}

    def training_epoch_end(self, outputs) -> None:
        # save chatbot datas to csv file
        # maby use db to store chatbot datas?
        original_embeddings = self.chatbot_embeddings
        updated_embeddings = torch.cat(self.embedding_list, dim=0).detach().cpu()
        new_embeddings = torch.cat((original_embeddings, updated_embeddings), dim=0).numpy()
        embeddings_dict = dict(enumerate(new_embeddings))
        chatbot_df = pd.DataFrame.from_dict({"answer": self.answer_dict,
                                             "embedding": embeddings_dict}) 
        if self.chatbot_embedding_out is None:
            raise Exception("Need 'chatbot_embedding_path' or 'chatbot_embedding_out' for training")
        Path(self.chatbot_embedding_out).parent.mkdir(parents=True, exist_ok=True)
        self._dump_chatbot_db(self.chatbot_embedding_out, chatbot_df)

    def predict_step(self, batch: Any, batch_idx) -> Any:
        sentence_embeddings = self.step(batch)
        score = self._batch_cos_sim(sentence_embeddings.detach().cpu(), self.chatbot_embeddings)
        best_score_idxes = np.argmax(score, axis=-1)
        print(best_score_idxes.shape)
        answers = []
        for idx in best_score_idxes:
            answers.append(self.answer_dict.get(int(idx)))
        return answers

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def configure_optimizers(self):
        return {
            "optimizer": self.hparams.optimizer(params=self.dummy_layer.parameters()),
        }

    def _batch_cos_sim(self, B, Z):
        B = B.T
        B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
        Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).

        # Distance matrix of size (b, n).
        cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
        return cosine_similarity

    def _dump_chatbot_db(self, filename: str, chatbot_df: pd.DataFrame):
        """_summary_

        Args:
            filename (str): parquet file path
            chatbot_df (pd.DataFrame): df to write
        """
        # parquet uses snappy compression engine for default
        # use gzip instead
        # speed: gizp < snappy, compression rate: gzip > snappy
        chatbot_df.to_parquet(filename, compression='snappy')
        
    def _read_chatbot_db(self, filename: str) -> pd.DataFrame:
        """Read chatobot_db

        Args:
            filename (str): parquet file path

        Returns:
            pd.DataFrame: chatbot_df from file
        """
        chatbot_df = pd.read_parquet(filename)
        return chatbot_df
        

        
if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    # cfg = omegaconf.OmegaConf.create({"paths": {"data_dir": "null", "output_dir": "logs/train/runs/2022-12-06_10-48-36/chatbot_db"},
    #                                   "model": "null"})
    # cfg.paths.data_dir = "/home/ubuntu/hanbin/2022-1-CECD3-EverySon-2/model/data/"
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "cos_sim_chatbot.yaml")
    model = hydra.utils.instantiate(cfg)
    model = CosSimChatbot.load_from_checkpoint("logs/train/runs/2022-12-06_10-48-36/checkpoints/last.ckpt")
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)
    
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

    print(tokenized_sentences)
    print(model.answer_dict)
    # y = model.predict_step(tokenized_sentences, batch_idx=0)
    # print(y)

    # model2 = CosSimChatbot.load_from_checkpoint("logs/train/runs/2022-12-06_10-48-36/checkpoints/last.ckpt")
    # print(model2.key)