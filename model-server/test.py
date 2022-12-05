from chatbot import Chatbot
import hydra
import omegaconf
import pyrootutils
import sys

sys.path.append("model/")
root = pyrootutils.setup_root(__file__, pythonpath=True)
cfg = omegaconf.OmegaConf.load(root / "model-server" / "model" / "configs" / "configs.yaml")

chatbot = Chatbot(cfg)

items = [{"speaker": "챗봇", "text": "오늘은 기분이 어때요?"}] #, {"speaker": "사용자", "text": "죽고 싶은 하루였어.."}]

class Chat:
    def __init__(self, speaker, text):
        self.speaker = speaker
        self.text = text


chat = Chat([0], ["죽고 싶은 하루였어.."])
print(chatbot.emotion_recognition(chat))
