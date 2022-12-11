# Ref.
# @article{lee2021compm,
#  title={CoMPM: Context Modeling with Speaker's Pre-trained Memory Tracking for Emotion Recognition in Conversation},
#  author={Lee, Joosung and Lee, Wooin},
#  journal={arXiv preprint arXiv:2108.11626},
#  year={2021}
# }
# some codes from https://github.com/rungjoo/compm
from typing import Any, Dict, Optional, Tuple, List
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
from typing import Union
import os
from transformers import AutoTokenizer

class CoMPM_Dataset(Dataset):
    def __init__(self, txt_file: str, dataclass: str='emotion'):
        """CoMPM Dataset. File format as Ref.

        Args:
            txt_file (str): Path of dataset file
            dataclass (str, optional): 'emotion' or 'sentiment'(Not using in our project). Defaults to 'emotion'.
        """
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []
        # 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sad', 'surprise'
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'joy': "joy", 'neutral': "neutral", 'sad': "sad", 'surprise': 'surprise'}
        self.sentidict = {'positive': ["joy"], 'negative': ["anger", "disgust", "fear", "sad"], 'neutral': ["neutral", "surprise"]}
        self.emoSet = set()
        self.sentiSet = set()
        for i, data in enumerate(dataset):
            if data == '\n':
                if len(self.dialogs) > 0:
                    self.speakerNum.append(len(temp_speakerList))
                    temp_speakerList = []
                    context = []
                    context_speaker = []
                continue
            speaker, utt, emo, senti = data.strip().split('\t')
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo], senti])
            self.emoSet.add(emodict[emo])
            self.sentiSet.add(senti)
        
        self.emoList = sorted(self.emoSet)  
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx: int) -> Tuple:
        """__getitem__

        Args:
            idx (int): idx

        Returns:
            Tuple: (dialogs (List):[context_speaker (List), context (List), emotion, sentiment],
                    labelList (List),
                    sentidict (Dict))
        """
        return self.dialogs[idx], self.labelList, self.sentidict

class CoMPMDataModule(LightningDataModule):
    """LightningDataModule for MEDIC dataset.

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
            data_dir: str = "data/",
            batch_size: int = 64,
            num_workers: int = 0,
            max_length: int = 256,
            train_val_test_split: Tuple[int, int, int] = None,
            pin_memory: bool = False,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_file = self.hparams.data_dir + '_train_ko.txt'
        self.val_file = self.hparams.data_dir + '_dev_ko.txt'
        self.test_file = self.hparams.data_dir + '_test_ko.txt'

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
            self.data_train = CoMPM_Dataset(self.train_file)
        if not self.data_val:
            self.data_val = CoMPM_Dataset(self.val_file)
        if not self.data_test:
            self.data_test = CoMPM_Dataset(self.test_file)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self._make_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._make_batch,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._make_batch,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def _encode_right_truncated(self, text, tokenizer, max_length=511):
        tokenized = tokenizer.tokenize(text)
        truncated = tokenized[-max_length:]    
        ids = tokenizer.convert_tokens_to_ids(truncated)
        
        return [tokenizer.cls_token_id] + ids

    def _padding(self, ids_list, tokenizer):
        max_len = 0
        for ids in ids_list:
            if len(ids) > max_len:
                max_len = len(ids)
        
        pad_ids = []
        for ids in ids_list:
            pad_len = max_len-len(ids)
            add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
            
            pad_ids.append(ids+add_ids)
        
        return torch.tensor(pad_ids)

    def _make_batch(self, sessions):
        batch_input, batch_labels, batch_speaker_tokens = [], [], []
        for session in sessions:
            data = session[0]
            label_list = session[1]
            
            context_speaker, context, emotion, sentiment = data
            now_speaker = context_speaker[-1]
            speaker_utt_list = []
            
            inputString = ""
            for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
                inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
                inputString += utt + " "
                
                if turn<len(context_speaker)-1 and speaker == now_speaker:
                    speaker_utt_list.append(self._encode_right_truncated(utt, self.tokenizer, self.hparams.max_length))
            
            concat_string = inputString.strip()
            batch_input.append(self._encode_right_truncated(concat_string, self.tokenizer, self.hparams.max_length))
            
            if len(label_list) > 3:
                label_ind = label_list.index(emotion)
            else:
                label_ind = label_list.index(sentiment)
            batch_labels.append(label_ind)        
            
            batch_speaker_tokens.append(self._padding(speaker_utt_list, self.tokenizer))
        
        batch_input_tokens = self._padding(batch_input, self.tokenizer)
        batch_labels = torch.tensor(batch_labels)    
        
        return {
            'input_token': batch_input_tokens,
            'label': batch_labels,
            'speaker_token': batch_speaker_tokens,
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "CoMPM_datamodule.yaml")
    cfg.data_dir = str(root / "data" / "MELD" / "multi") + "/MELD"
    cfg.pretrained_model_name_or_path = "klue/bert-base"
    data = hydra.utils.instantiate(cfg)
    data.setup()
    train_dataloader = data.train_dataloader()
    for x in train_dataloader:
        print(x)
        break
    train_dataset = CoMPM_Dataset(cfg.data_dir + "MELD_train_ko.txt")
    print(len(train_dataset))
    print(len(train_dataloader))