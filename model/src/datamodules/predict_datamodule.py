from typing import Any, Dict, Optional, Tuple, List
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from typing import Union
import os
from transformers import AutoTokenizer
from src.datamodules.components.bert_datamodule import BERTDataset, BERTDataModule


class PredictDataset(BERTDataset):
    pass


class PredictDataModule(BERTDataModule):
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

    def __init__(
            self,
            pretrained_model_name_or_path: str,
            num_labels: int,
            class_names: List[str],
            data_dir: str,
            batch_size: int = 64,
            num_workers: int = 0,
            max_length: int = 256,
            pin_memory: bool = False,
    ):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            num_labels=num_labels,
            class_names=class_names,
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
            pin_memory=pin_memory,
        )
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_predict: Optional[Dataset] = None

    def predict_dataloader(self):
        self.setup()
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if self.data_predict is None:
            df = pd.read_csv(self.hparams.data_dir, encoding="UTF-8")
            tokenized_sentences = self.tokenizer(
                df["text"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
                max_length=self.hparams.max_length,)

            self.data_predict = PredictDataset(encodings=tokenized_sentences)

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "emotion_predict.yaml")
    cfg.data_dir = str(root / "data" / "data.csv")
    cfg.pretrained_model_name_or_path = "klue/bert-base"
    data = hydra.utils.instantiate(cfg)
    data.setup()
    predict_dataloader = data.predict_dataloader()
    for x in predict_dataloader:
        print(x)
        break


