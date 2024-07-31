import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from conversation import load_and_cache_examples, collate_fn, eval_batch_size, device

def evaluate(model, tokenizer, df_trn, df_val):
    eval_dataset = load_and_cache_examples(tokenizer, df_trn, df_val, evaluate=True)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), 
                                 batch_size=eval_batch_size, collate_fn=collate_fn)
 
    eval_loss = 0.0
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, attention_masks = batch
        inputs = inputs.to(device)
        attention_masks = attention_masks.to(device)
        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_masks, labels=inputs)
            eval_loss += outputs.loss.mean().item()

    eval_loss /= len(eval_dataloader)
    perplexity = torch.exp(torch.tensor(eval_loss))
    return {"eval_loss": eval_loss, "perplexity": perplexity}