import hydra
import torch
import torch.nn as nn

from .src.compm import *

class_names=[
    "분노",
    "혐오",
    "공포",
    "행복",
    "중립",
    "슬픔",
    "놀람",
]

class Chatbot:
    def __init__(self, cfg):
        # self.emotion_recognition_model = hydra.utils.instantiate(cfg.emotion)
        # self.cos_sim_model = hydra.utils.instantiate(cfg.cos_sim)
        self.emotion_recognition_model = None
        self.tokenizer = None
        self.cos_sim_model = None
        self.gpt = None
        self.gpt_tokenizer = None
        self.load_model(cfg)

    def load_model(self, cfg):
        # self.gpt_tokenizer = hydra.utils.instantiate(cfg.gpt_tokenizer)
        # self.gpt = hydra.utils.instantiate(
        #     cfg.gpt, 
        #     pad_token_id=self.gpt_tokenizer.eos_token_id).to(device='cuda', non_blocking=True)
        # self.gpt.eval()

        self.tokenizer = hydra.utils.instantiate(cfg.tokenizer)
        self.emotion_recognition_model = hydra.utils.instantiate(cfg.emotion_recognition)
        self.emotion_recognition_model.eval()
        

    # def generate_chat(self, chats):
    #     prompt = self._read_prompt()
    #     chats = [f"{chat['speaker']}:{chat['text']}" for chat in chats]
    #     chats = "\n".join(chats).strip()
    #     prompt += "\n" + chats + "\n챗봇:"
        
    #     with torch.no_grad():
    #         tokens = self.gpt_tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
    #         gen_tokens = self.gpt.generate(tokens, do_sample=True, temperature=0.7, max_new_tokens=50,
    #                                     early_stopping=True, eos_token_id=63997)

    #     generated = self.gpt_tokenizer.batch_decode(gen_tokens)[0].strip()
    #     generated = generated.split("###")[-1].strip().split("\n")[-1]
    #     generated = generated.replace("챗봇:", "").strip()
    #     return generated
        
    # def _read_prompt(self, task="suicide"):
    #     with open(f"chatbot/prompt/{task}.txt", "r") as f:
    #         prompt = f.readlines()
    #     prompt = "".join(prompt).strip()
    #     return prompt
    
    def emotion_recognition(self, chat):
        input_token, speaker_token = preprocess(chat.speaker, chat.text, self.tokenizer)
        outputs = self.emotion_recognition_model.forward(input_token.cuda(), speaker_token.cuda())
        y = outputs.detach().cpu().squeeze().argmax(dim=-1)
        label = class_names[y]
        return {"label": label}
