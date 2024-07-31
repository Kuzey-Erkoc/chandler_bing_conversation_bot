from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from conversation import collate_fn, train_batch_size, device, lr, steps, output_dir

def train(train_dataset, model, tokenizer):
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), 
                                  batch_size=train_batch_size, collate_fn=collate_fn)

    total_training = len(train_dataloader) * 3

    optimizer = AdamW(model.parameters(), lr=lr, no_deprecation_warning=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
                                                num_training_steps=total_training)

    global_step = 0
    model.zero_grad()
    losses = []
    
    for epoch in range(3):
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            inputs, attention_masks = batch
            inputs = inputs.to(device)
            attention_masks = attention_masks.to(device)
            model.train()
            outputs = model(inputs, attention_mask=attention_masks, labels=inputs)
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if global_step % steps == 0:
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        losses.append(avg_epoch_loss)
        print(f"Average loss for epoch {epoch+1}: {avg_epoch_loss:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return global_step, losses
 