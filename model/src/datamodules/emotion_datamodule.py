from typing import Any, Dict, Optional, Tuple, List
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import pandas as pd
from typing import Union
import os
from transformers import AutoTokenizer
from src.datamodules.components.bert_datamodule import BERTDataset, BERTDataModule


class EMOTIONDataset(BERTDataset):
    pass


class EMOTIONDataModule(BERTDataModule):
    """LightningDataModule for EMOTION dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            df = pd.read_csv(self.hparams.data_dir, encoding="UTF-8")
            tokenized_sentences = self.tokenizer(
                df["text"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
                max_length=self.hparams.max_length,)

            labels = df["label"].values
            ratio = self.hparams.train_val_test_split
            dlen = len(df)
            lengths = [
                int(dlen * ratio[0]/sum(ratio)),
                int(dlen * ratio[1]/sum(ratio)),
                dlen - int(dlen * ratio[0]/sum(ratio)) - int(dlen * ratio[1]/sum(ratio))
            ]
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=EMOTIONDataset(encodings=tokenized_sentences, labels=labels),
                lengths=lengths,
                generator=torch.Generator().manual_seed(42),
            )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "emotion.yaml")
    cfg.data_dir = str(root / "data" / "data.csv")
    cfg.pretrained_model_name_or_path = "klue/bert-base"
    data = hydra.utils.instantiate(cfg)
    data.setup()
    train_dataloader = data.train_dataloader()
    for x in train_dataloader:
        print(x)
        break