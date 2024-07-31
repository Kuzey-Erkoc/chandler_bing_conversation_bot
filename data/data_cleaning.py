import pandas as pd
df = pd.read_csv("data/friends_quotes.csv")
df = df.drop(["episode_number", "episode_title", "quote_order", "season"], axis=1)[(df["author"] == "Chandler") & 
                                                                                   (~df["quote"].str.contains(r"\(.*?\)", regex=True)) & 
                                                                                   (df["quote"].str.split().str.len() >= 3)].to_csv("data/cleaned_dataset.csv")
