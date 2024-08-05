import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters and configuration
random_seed = 42
train_batch_size = 4
eval_batch_size = 4
lr = 5e-5
steps = 3500
output_dir = "output_dir"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Data loading and preprocessing
chandler_bing_lines_dataset = pd.read_csv("data/friends_quotes.csv")
number_of_examples = 7

contexted = [[chandler_bing_lines_dataset["quote"][j] for j in range(i, i - number_of_examples - 1, -1)] 
             for i in range(number_of_examples, len(chandler_bing_lines_dataset["quote"]))]

columns = ["response"] + [f"context/{i}" for i in range(number_of_examples)]
dataset = pd.DataFrame(contexted, columns=columns)
train_df, validation_df = train_test_split(dataset, test_size=0.1)

def construct_conv(row, tokenizer):
    conv = [tokenizer.encode(x, add_special_tokens=True) + [tokenizer.eos_token_id] for x in row]
    input_ids = [item for sublist in conv for item in sublist]
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask

class ConversationDataset(Dataset):
    def __init__(self, tokenizer, df):
        self.examples = []
        self.attention_masks = []
        for _, row in df.iterrows():
            input_ids, attention_mask = construct_conv(row, tokenizer)
            self.examples.append(input_ids)
            self.attention_masks.append(attention_mask)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long), torch.tensor(self.attention_masks[item], dtype=torch.long)

def load_and_cache_examples(tokenizer, df_trn, df_val, evaluate=False):
    return ConversationDataset(tokenizer, df_val if evaluate else df_trn)

def set_seed():
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return padded_input_ids, padded_attention_masks
