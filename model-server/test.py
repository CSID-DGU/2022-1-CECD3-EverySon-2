from chatbot import Chatbot
import hydra
import omegaconf
import pyrootutils

root = pyrootutils.setup_root(__file__, pythonpath=True)
cfg = omegaconf.OmegaConf.load(root / "model-server" / "model" / "configs" / "configs.yaml")

chatbot = Chatbot(cfg)

items = [{"speaker": "챗봇", "text": "오늘은 기분이 어때요?"}] #, {"speaker": "사용자", "text": "죽고 싶은 하루였어.."}]

print("챗봇: 오늘은 기분이 어때요?")
for i in range(3):
    text = input("사용자:")
    item = {"speaker": "사용자", "text": text}
    items.append(item)
    gen = chatbot.generate_chat(items)
    print(gen)
    items.append({"speaker": "챗봇", "text": gen})
    items = items[-1:]
    print(items)
    
