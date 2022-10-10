from src.datamodules.components.tucore_gcn_component import TUCOREGCNDataset, TUCOREGCNDataloader

from typing import Any, Dict, Optional, Tuple, List
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset
import os
from transformers import AutoTokenizer


class TUCOREGCNDataModule(LightningDataModule):
    """LightningDataModule for TUCORE-GCN dataset.

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
            src_file,
            pretrained_model_name_or_path: str,
            num_labels: int,
            encoder_type: str = "BERT",
            max_length: int = 512,
            window_size: int = 1,
            batch_size: int = 8,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model_name_or_path)

    @property
    def num_classes(self):
        return self.hparams.num_labels

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train:
            self.data_train = TUCOREGCNDataset(
                src_file=self.hparams.src_file,
                save_file=self.hparams.src_file + "/train_" + self.hparams.encoder_type + ".pkl",
                max_length=self.hparams.max_length,
                tokenizer=self.tokenizer,
                n_class=self.hparams.num_labels,
                encoder_type=self.hparams.encoder_type,
                window_size=self.hparams.window_size,
            )

        if not self.data_val:
            self.data_val = TUCOREGCNDataset(
                src_file=self.hparams.src_file,
                save_file=self.hparams.src_file + "/dev_" + self.hparams.encoder_type + ".pkl",
                max_length=self.hparams.max_length,
                tokenizer=self.tokenizer,
                n_class=self.hparams.num_labels,
                encoder_type=self.hparams.encoder_type,
                window_size=self.hparams.window_size,
            )

        if not self.data_test:
            self.data_test = TUCOREGCNDataset(
                src_file=self.hparams.src_file,
                save_file=self.hparams.src_file + "/test_" + self.hparams.encoder_type + ".pkl",
                max_length=self.hparams.max_length,
                tokenizer=self.tokenizer,
                n_class=self.hparams.num_labels,
                encoder_type=self.hparams.encoder_type,
                window_size=self.hparams.window_size,
            )

    def train_dataloader(self):
        return TUCOREGCNDataloader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            relation_num=self.hparams.num_labels,
            max_length=self.hparams.max_length,
        )

    def val_dataloader(self):
        return TUCOREGCNDataloader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            relation_num=self.hparams.num_labels,
            max_length=self.hparams.max_length,
        )

    def test_dataloader(self):
        return TUCOREGCNDataloader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            relation_num=self.hparams.num_labels,
            max_length=self.hparams.max_length,
        )


if __name__ == "__main__":
    datamodule = TUCOREGCNDataModule(
        src_file="../../data/MELD",
        pretrained_model_name_or_path="klue/roberta-base",
        num_labels=7,
    )
    datamodule.setup()