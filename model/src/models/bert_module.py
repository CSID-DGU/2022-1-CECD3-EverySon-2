from typing import Any, List
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from wandb.plot import confusion_matrix
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn
from typing import Optional


class BertForClassification(LightningModule):
    """
    LightningModule for BERT classification.
    """

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            class_names,
            num_labels: int,
            hidden_dropout_prob: float = 0.1,
            hidden_size: int = 768,
            freeze: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        if freeze is True:
            self._freeze_layer(self.net)

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self._init_weights(self.classifier)
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # accuracy
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # f1-score
        self.train_f1 = F1Score(num_classes=num_labels, average="macro")
        self.val_f1 = F1Score(num_classes=num_labels, average="macro")
        self.test_f1 = F1Score(num_classes=num_labels, average="macro")

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

    def _freeze_layer(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, **kwargs: Any):
        outputs = self.net(**kwargs)
        # hidden_states = outputs[1]
        # pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
        # pooled_output = pooled_output[:, 0, :]
        # pooled_output = self.dropout(pooled_output)
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        return logits

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure {metric}_best doesn't store {metric} from these checks
        self.val_acc_best.reset()
        self.val_f1_best.reset()

    def step(self, batch: Any):
        y = batch.pop("label")
        logits = self.forward(**batch)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_index: int):
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
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "bert.yaml")
    model = hydra.utils.instantiate(cfg)
    input_ids = torch.zeros(1, 512, dtype=torch.long)
    attention_mask = torch.zeros(1, 512, dtype=torch.long)
    token_type_ids = torch.zeros(1, 512, dtype=torch.long)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
    y = model.forward(**batch)
    print(y)
