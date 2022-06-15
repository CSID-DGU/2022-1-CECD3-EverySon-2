from typing import Any, List

import numpy as np
import torch
import torch.optim as optim
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MaxMetric
import pytorch_lightning as pl
from wandb.plot import confusion_matrix
from transformers import AutoTokenizer, BertForSequenceClassification, ElectraForSequenceClassification, AlbertForSequenceClassification
from transformers import AlbertForSequenceClassification, RobertaForSequenceClassification

class BERTLitModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str,
        output_size: int,
        lr: float = 5e-5,
        lr_scheduler: str = "",
        optimizer: str = "AdamW",
        class_names: list = [],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if "albert" in self.hparams.pretrained_model:
            self.model = AlbertForSequenceClassification.from_pretrained(self.hparams.pretrained_model, num_labels=self.hparams.output_size)
        elif "roberta" in self.hparams.pretrained_model:
            self.model = RobertaForSequenceClassification.from_pretrained(self.hparams.pretrained_model, num_labels=self.hparams.output_size)
        elif "electra" in self.hparams.pretrained_model:
            self.model = ElectraForSequenceClassification.from_pretrained(self.hparams.pretrained_model, num_labels=self.hparams.output_size)
        elif "bert" in self.hparams.pretrained_model:
            self.model = BertForSequenceClassification.from_pretrained(self.hparams.pretrained_model, num_labels=self.hparams.output_size)
        else:
            raise ValueError(f"{self.hparams.pretrained_model} is not supported.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
    
        # loss function
        # self.criterion = torch.nn.CrossEntropyLoss()
        
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()


    def forward(self, **kwargs: Any):
        return self.model(**kwargs)

    def step(self, batch: Any):
        outputs = self.forward(**batch)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        return loss, preds, batch["labels"]

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"preds": preds.detach().cpu(), "labels": targets.detach().cpu()}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        
        labels = torch.cat([output["labels"] for output in outputs])
        preds = torch.cat([output["preds"] for output in outputs])
        
        # confusion_matrix
        self.logger.experiment[0].log({"conf_mat": confusion_matrix(
                            probs=None, y_true=labels.numpy(), preds=preds.numpy(),
                            class_names=self.hparams.class_names)}
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"preds": preds.detach().cpu(), "labels": targets.detach().cpu()}

    def test_epoch_end(self, outputs: List[Any]):
        labels = torch.cat([output["labels"] for output in outputs])
        preds = torch.cat([output["preds"] for output in outputs])

        # confusion_matrix
        self.logger.experiment[0].log({"conf_mat": confusion_matrix(
                            probs=None, y_true=labels.numpy(), preds=preds.numpy(),
                            class_names=self.hparams.class_names)}
        )

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def predict_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        return preds

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)
        if self.hparams.lr_scheduler == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        elif self.hparams.lr_scheduler == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(optimizer)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }