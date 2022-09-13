from typing import Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import os
import pandas as pd
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class TextDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir: str,
            block_size: int,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        if os.path.isfile(data_dir) is False:
            raise ValueError(f"Input file path {data_dir} not found")

        lines = pd.read_csv(data_dir, encoding="UTF-8")
        lines = lines["text"].tolist()
        batch_encoding = tokenizer(lines,
                                   add_special_tokens=True,
                                   truncation=True,
                                   max_length=block_size,
        )
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class MaskedLMDataModule(LightningDataModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        max_length: int = 512,
        pin_memory: bool = False,
        mlm=True,
        mlm_probability: float = 0.15
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model_name_or_path)
        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
        )

    def setup(self, stage: Optional[str] = None):
        if not self.data_train:
            self.data_train = TextDataset(
                tokenizer=self.tokenizer,
                data_dir=self.hparams.data_dir,
                block_size=self.hparams.max_length,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            collate_fn=self.collate_fn,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )


