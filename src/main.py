import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from conversation import set_seed, load_and_cache_examples, output_dir, device
from train import train
from evaluate import evaluate
from plot_metrics import plot_training_loss

def main(df_trn, df_val):
    set_seed()

    model_name = "microsoft/DialoGPT-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    train_dataset = load_and_cache_examples(tokenizer, df_trn, df_val)
    global_step, losses = train(train_dataset, model, tokenizer)
    plot_training_loss(losses)

    result = evaluate(model, tokenizer, df_trn, df_val)
    print(result)
    
    with open(os.path.join(output_dir, "eval_results.txt"), "w") as f:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")

class Conversation:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_history_ids = None

    def chat(self, user_input):
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt")
        
        bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1) if self.chat_history_ids is not None else new_user_input_ids
        self.chat_history_ids = self.model.generate(
            bot_input_ids, 
            max_length=300,
            pad_token_id=self.tokenizer.eos_token_id,  
            no_repeat_ngram_size=3,       
            do_sample=True, 
            top_k=100, 
            top_p=0.7,
            temperature=0.8
        )
        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
                                         skip_special_tokens=True)
        return response

def load_saved_model(model_path):
    print(f"Loading saved model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    return model, tokenizer
