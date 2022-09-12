
# import pandas as pd
# from tqdm import tqdm
# with open("data/data.csv", "r") as f:
#     lines = f.readlines()
# df = pd.DataFrame([], columns=["text", "label"])

# for i, line in tqdm(enumerate(lines)):
#     tmp = line.split(",")
#     label = tmp[-1]
#     text = tmp[:-1]
#     text = ",".join(text)
#     df.loc[i] = [text.strip(), label.strip()]

# print(df.head())
# df.to_csv("data/predict.csv", encoding="UTF-8", index=False)
import pandas as pd

df = pd.read_csv("data/data.csv")
print(len(df))
df = df.drop_duplicates(subset=["text"])
df.to_csv("data/data.csv", encoding="UTF-8", index=False)
print(df.head())
print(len(df))