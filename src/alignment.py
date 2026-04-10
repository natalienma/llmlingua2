# continuous scoring from attention
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

def attention_scores(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.attentions: tuple of (num_layers, batch, heads, seq, seq)
    # average across all layers and heads
    attentions = torch.stack(outputs.attentions)  # [12, 1, 12, seq, seq]
    avg_attention = attentions.mean(dim=(0, 1, 2))  # [seq, seq]
    
    # importance = how much attention each token receives from all others
    token_importance = avg_attention.mean(dim=0)  # [seq]
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return list(zip(tokens, token_importance.tolist()))

scores = attention_scores("The Great Depression was a severe global economic crisis.")
for token, score in scores:
    print(f"{token}: {score:.3f}")