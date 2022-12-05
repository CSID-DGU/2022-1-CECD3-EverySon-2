from typing import List, Dict, Any
import torch
import torch.nn as nn
import sys

def _encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]
    ids = tokenizer.convert_tokens_to_ids(truncated)

    return [tokenizer.cls_token_id] + ids


def _padding(ids_list, tokenizer):
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


def preprocess(speaker, text, tokenizer):
    context_speaker, context = speaker, text
    now_speaker = context_speaker[-1]
    speaker_utt_list = []
    inputString = ""
    for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
        inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
        inputString += utt + " "
        if turn<len(context_speaker)-1 and speaker == now_speaker:
            speaker_utt_list.append(_encode_right_truncated(utt, tokenizer, 512))

    concat_string = inputString.strip()

    data_dict = {}

    data_dict["input_token"] = torch.tensor(_encode_right_truncated(concat_string, tokenizer, 512))
    data_dict["speaker_token"] = _padding(speaker_utt_list, tokenizer)
    return data_dict["input_token"].unsqueeze(0), data_dict["speaker_token"].unsqueeze(0)