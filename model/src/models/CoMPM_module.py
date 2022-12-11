# Ref.
# @article{lee2021compm,
#  title={CoMPM: Context Modeling with Speaker's Pre-trained Memory Tracking for Emotion Recognition in Conversation},
#  author={Lee, Joosung and Lee, Wooin},
#  journal={arXiv preprint arXiv:2108.11626},
#  year={2021}
# }
# some codes from https://github.com/rungjoo/compm

from functools import lru_cache
from sqlite3 import paramstyle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pandas as pd
import pdb

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model

from transformers import RobertaConfig, BertConfig

from transformers import AutoModel
from typing import Dict, Union, Any, List

from torchmetrics import MaxMetric
from torchmetrics import Accuracy, F1Score, ConfusionMatrix

from pytorch_lightning import LightningModule

import gc

class ERC_model(nn.Module):
    def __init__(self,
                 clsNum,
                 last: bool=False,
                 freeze: bool=False,
                 pretrained_model_name_or_path: Union[str, os.PathLike] = "klue/bert-base",
                 ):
        super(ERC_model, self).__init__()
        self.gpu = True
        self.last = last
        
        """Model Setting"""
        self.context_model = AutoModel.from_pretrained(pretrained_model_name_or_path)

        self.speaker_model = AutoModel.from_pretrained(pretrained_model_name_or_path)

        self.hiddenDim = self.context_model.config.hidden_size
        
        zero = torch.empty(2, 1, self.hiddenDim).cuda()
        self.h0 = torch.zeros_like(zero) # (num_layers * num_directions, batch, hidden_size)
        self.speakerGRU = nn.GRU(self.hiddenDim, self.hiddenDim, 2, dropout=0.3) # (input, hidden, num_layer) (BERT_emb, BERT_emb, num_layer)
            
        """score"""
        # self.SC = nn.Linear(self.hiddenDim, self.hiddenDim)
        self.W = nn.Linear(self.hiddenDim, clsNum)
        
        """parameters"""
        self.train_params = list(self.context_model.parameters())+list(self.speakerGRU.parameters())+list(self.W.parameters()) # +list(self.SC.parameters())
        if not freeze:
            self.train_params += list(self.speaker_model.parameters())

    def forward(self, batch_input_tokens, batch_speaker_tokens):
        """
            batch_input_tokens: (batch, len)
            batch_speaker_tokens: [(speaker_utt_num, len), ..., ]
        """
        if self.last:   # gpt-2
            batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:,-1,:] # (batch, 1024)
        else:
            batch_context_output = self.context_model(batch_input_tokens).last_hidden_state[:,0,:] # (batch, 1024)
        
        batch_speaker_output = []
        for speaker_tokens in batch_speaker_tokens:
            if speaker_tokens.shape[0] == 0:
                speaker_track_vector = torch.zeros(1, self.hiddenDim).cuda()
            else:
                if self.last:
                    speaker_output = self.speaker_model(speaker_tokens.cuda()).last_hidden_state[:,-1,:] # (speaker_utt_num, 1024)
                else:
                    speaker_output = self.speaker_model(speaker_tokens.cuda()).last_hidden_state[:,0,:] # (speaker_utt_num, 1024)
                speaker_output = speaker_output.unsqueeze(1) # (speaker_utt_num, 1, 1024)
                speaker_GRU_output, _ = self.speakerGRU(speaker_output, self.h0) # (speaker_utt_num, 1, 1024) <- (seq_len, batch, output_size)
                speaker_track_vector = speaker_GRU_output[-1,:,:] # (1, 1024)
            batch_speaker_output.append(speaker_track_vector)
        batch_speaker_output = torch.cat(batch_speaker_output, 0) # (batch, 1024)
                   
        final_output = batch_context_output + batch_speaker_output
        # final_output = batch_context_output + self.SC(batch_speaker_output)        
        context_logit = self.W(final_output) # (batch, clsNum)
        
        return context_logit


class CoMPM(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        # scheduler: torch.optim.lr_scheduler,
        class_names,
        num_labels: int,
        freeze: bool = True,
        *args, **kwargs
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net

        if freeze is True:
            self._freeze_layer(self.net)

        self.criterion = torch.nn.CrossEntropyLoss()

        # accuracy
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # f1-score
        self.train_f1 = F1Score(num_classes=num_labels, average="weighted")
        self.val_f1 = F1Score(num_classes=num_labels, average="weighted")
        self.test_f1 = F1Score(num_classes=num_labels, average="weighted")

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()
    
    def forward(self, input_token, speaker_token):
        return self.net(input_token, speaker_token)

    def step(self, batch: Dict):
        input_token = batch["input_token"]
        pred_logits = self.forward(batch["input_token"], batch["speaker_token"])
        loss_val = self.criterion(pred_logits, batch["label"])
        pred_labels = torch.argmax(pred_logits, dim=1)
        return loss_val, pred_labels, batch["label"]

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure {metric}_best doesn't store {metric} from these checks
        self.val_acc_best.reset()
        self.val_f1_best.reset()

    def training_step(self, batch: Dict, batch_idx):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        f1 = self.train_f1(preds, targets)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/acc", acc, prog_bar=True, sync_dist=True)
        self.log("train/f1", f1, prog_bar=True, sync_dist=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}
        
    def training_step_end(self, outputs):
        gc.collect()
        torch.cuda.empty_cache()
        return outputs["loss"]

    def train_epoch_end(self, outputs: List[Any]):
        self.train_acc.reset()
        self.train_f1.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        f1 = self.val_f1(preds, targets)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/f1", f1, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()
        f1 = self.val_f1.compute()
        self.val_acc_best.update(acc)
        self.val_f1_best.update(f1)
        self.val_acc.reset()
        self.val_f1.reset()

        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        f1 = self.test_f1(preds, targets)
        self.log("test/loss", loss, on_epoch=True, sync_dist=True)
        self.log("test/acc", acc, on_epoch=True, sync_dist=True)
        self.log("test/f1", f1, on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs: List[Any]):
        self.test_acc.reset()
        self.test_f1.reset()

    def predict_step(self, batch: Any, batch_idx, dataloader_idx=0):
        logits = self.forward(**batch)
        preds = torch.argmax(logits, dim=-1).cpu().detach()
        return preds
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        Todo: 
            Fix scheduler problem. While using get_linear_schedule_with_warmup, as paper, 
            seems model dosen't train well.
            Set train_steps from dataloader
            (train_steps Ref.
                https://github.com/Lightning-AI/lightning/issues/1038#issuecomment-603930802
                https://github.com/Lightning-AI/lightning/discussions/10652
                https://github.com/Lightning-AI/lightning/issues/10430)
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        # # Todo: set train_steps from dataloader
        # # self.trainer.reset_train_dataloader()
        # # train_steps = len(self.train_dataloader) * self.hparams.max_epochs
        # train_steps = self.hparams.len_train_dataloader
        # scheduler = self.hparams.scheduler(optimizer=optimizer,
        #                                    num_warmup_steps=int(self.hparams.warmup_ratio * train_steps),
        #                                    num_training_steps=train_steps)
        return {
            "optimizer": optimizer,
            # "scheduler": scheduler,
        }