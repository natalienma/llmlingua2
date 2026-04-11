from datasets import load_dataset
from scipy.spatial.distance import cosine
from compress import compress_with_gpt
from reconstruct import embed
import tiktoken
import json

ds = load_dataset("rajpurkar/squad", split="train", streaming=True)
enc = tiktoken.encoding_for_model("gpt-4")

# 100 unique
seen_passages = {}
for item in ds:
    passage = item['context']
    if passage not in seen_passages:
        seen_passages[passage] = item
    if len(seen_passages) >= 100:
        break

og_compressed = {}
for passage in seen_passages:
    compressed = compress_with_gpt(passage)
    passage_embed = embed(passage)
    compressed_embed = embed(compressed)
    og_compressed[passage] = {
        "compressed": compressed,
        "cosine": 1 - cosine(passage_embed, compressed_embed),
        "ratio": len(enc.encode(passage)) / len(enc.encode(compressed))
    }

dataset = []
for passage, item in seen_passages.items():
    entry = og_compressed[passage]
    dataset.append({
        "passage": passage,
        "compressed_passage": entry["compressed"],
        "question": item["question"],
        "answer": item["answers"]["text"][0],
        "compression_ratio": entry["ratio"],
        "cosine_score": entry["cosine"],
        "id": item["id"]
    })

print(dataset[0])

with open("squad_compressed.json", "w") as f:
    json.dump(dataset, f, indent=2)