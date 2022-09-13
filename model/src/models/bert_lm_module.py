from typing import Any
import torch
from pytorch_lightning import LightningModule


class BertForMaskedLM(LightningModule):
    """
    LightningModule for Bert Masked Language Modeling
    """
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

    def forward(self, **inputs):
        outputs = self.net(**inputs)
        return outputs

    def step(self, batch: Any):
        """
        return : MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        """
        y = batch["labels"]
        outputs = self.forward(**batch)
        return outputs.loss, outputs.logits, y

    def training_step(self, batch: Any, batch_index: int):
        loss, _, _ = self.step(batch)
        return {"loss": loss}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }








