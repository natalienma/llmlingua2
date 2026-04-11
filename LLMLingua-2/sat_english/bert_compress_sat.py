import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

# Load trained model
model_path = "/content/drive/MyDrive/bert-compressor"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

def bert_compress(text, target_ratio=3.0):
    words = text.split()
    if not words:
        return text
    
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    word_ids = inputs.word_ids()
    
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits[0], dim=-1)
    keep_probs = probs[:, 1]
    
    # map subword probs to word probs
    word_scores = {}
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        score = keep_probs[token_idx].item()
        if word_id not in word_scores or score > word_scores[word_id]:
            word_scores[word_id] = score
    
    # keep top tokens to hit target ratio
    n_keep = max(1, int(len(words) / target_ratio))
    sorted_ids = sorted(word_scores, key=word_scores.get, reverse=True)
    keep_set = set(sorted_ids[:n_keep])
    
    compressed = " ".join(words[i] for i in range(len(words)) if i in keep_set)
    return compressed

# Load SAT data and compress
with open("sat_compressed.json") as f:
    data = json.load(f)

bert_results = []
for item in data:
    bert_comp = bert_compress(item['passage'], target_ratio=3.0)
    bert_results.append({
        **item,
        "bert_compressed": bert_comp
    })

with open("sat_bert_compressed.json", "w") as f:
    json.dump(bert_results, f, indent=2)

print("Done")