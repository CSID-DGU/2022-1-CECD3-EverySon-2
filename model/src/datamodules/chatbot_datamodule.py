from typing import Any, Dict, Optional, Tuple, List
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from typing import Union
import os
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class ChatbotDataset(Dataset):
    def __init__(self, encodings, answers=None):
        super().__init__()
        self.encodings = encodings
        self.answers=answers

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        if self.answers is not None:
            item["answer"] = self.answers[index]
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class ChatbotDataModule(LightningDataModule):
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
            train_data_dir: str = None,
            pred_data_dir: str = None,
            batch_size: int = 64,
            num_workers: int = 0,
            max_length: int = 256,
            pin_memory: bool = False,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model_name_or_path)

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
        if self.data_train is None:
            # - text   : user input text
            # - answer : chatbot response
            # - label(optional)  : emotion label
            df = pd.read_csv(self.hparams.train_data_dir, encoding='UTF-8')
            if "label" in df.columns:
                tokenizer_inputs = (df["text"] + " [SEP] " + df["label"]).tolist()
            else:
                tokenizer_inputs = df["text"].tolist()
            tokenized_sentences = self.tokenizer(
                tokenizer_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
                max_length=self.hparams.max_length,)
            self.data_train = ChatbotDataset(encodings=tokenized_sentences, answers=df["answer"].tolist())

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def pred_setup(self):
        # Resolve WARNING:
        # huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
        # To disable this warning, you can either:
        #         - Avoid using `tokenizers` before the fork if possible
        #         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.data_predict is None:
            df = pd.read_csv(self.hparams.pred_data_dir, encoding="UTF-8")
            tokenized_sentences = self.tokenizer(
                (df["text"] + " [SEP] " + df["label"]).tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
                max_length=self.hparams.max_length,)
            self.data_predict = ChatbotDataset(encodings=tokenized_sentences)

    def predict_dataloader(self):
        self.pred_setup()
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "chatbot_predict.yaml")
    cfg.data_dir = str(root / "data" / "data.csv")
    cfg.pretrained_model_name_or_path = "jhgan/ko-sroberta-multitask"
    data = hydra.utils.instantiate(cfg)
    data.setup()
    predict_dataloader = data.predict_dataloader()
    for x in predict_dataloader:
        print(x)
        break


