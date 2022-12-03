import hydra
import torch
import torch.nn as nn
class Chatbot:
    def __init__(self, cfg):
        self.emotion_recognition_model = hydra.utils.instantiate(cfg.emotion)
        # self.cos_sim_model = hydra.utils.instantiate(cfg.cos_sim)
        self.nlg =
