from typing import Optional, Tuple
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from transformers import AutoTokenizer

class KocDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index])
        return item
    
    def __len__(self):
        return len(self.encodings["input_ids"])


class KocDataModule(LightningDataModule):
    def __init__(
        self,
        pretrained_model: str,
        data_train: str,
        data_val: str,
        data_test: str,
        num_classes: int,
        batch_size: int = 64,
        max_length: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        class_names: list = [],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        # self.data_predict: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def preprocess(self, df):
        df.dropna(axis=0)
        return df

    def load_data(self, path, mode):
        df = pd.read_csv(path, encoding="utf-8")
        emotion_map = {"행복":0, "슬픔":1, "놀람":2, "분노":3, "공포":4, "혐오":5, "중립":6}
        df = self.preprocess(df)
        tokenized_sentences = self.tokenizer(
                                                list(df["Sentence"][0:]),
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                add_special_tokens=True,
                                                max_length=self.hparams.max_length,
                            )
        if mode in ["train", "val", "test"]:
            df["Emotion"] = df["Emotion"].map(emotion_map)
            labels = df["Emotion"].values
            return KocDataset(encodings=tokenized_sentences, labels=labels)
        else:
            return KocDataset(encodings=tokenized_sentences, labels=None)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = self.load_data(path=self.hparams.data_train, mode="train")
            self.data_val = self.load_data(path=self.hparams.data_val, mode="val")
            self.data_test = self.load_data(path=self.hparams.data_test, mode="test")
            
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )