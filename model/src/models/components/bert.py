from transformers import AutoModel, AutoModelForMaskedLM
import torch.nn as nn
from typing import Union, Any
import os


class BertModel(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Union[str, os.PathLike] = "klue/bert-base",
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path)

    def forward(self, **kwargs: Any):
        return self.model(**kwargs)


class BertMaskedLM(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Union[str, os.PathLike] = "klue/bert-base",
    ):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path)

    def forward(self, **inputs: Any):
        return self.model(**inputs)


if __name__ == "__main__":
    net = BertModel()
