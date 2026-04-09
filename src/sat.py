from datasets import load_dataset
from scipy.spatial.distance import cosine
from compress import compress_with_gpt
from reconstruct import embed
import tiktoken
import json 

ds = load_dataset("emozilla/sat-reading")
enc = tiktoken.encoding_for_model("gpt-4")

def separate(ds): #split all passages and questions
    passages = []
    questions = []
    # somehow create a new category so there are 5: question, passage, answer, requires_line, id
    for item in ds['train']:
        parts = item['text'].split("Question")
        passages.append(parts[0])
        questions.append(parts[1])
    return passages, questions

passages, questions = separate(ds)

# create dictionary to get rid of repeating passages
og_compressed = {}  # passage -> {compressed, cosine, ratio}

for passage in passages:
    if passage not in og_compressed:
        compressed = compress_with_gpt(passage)
        passage_embed = embed(passage)
        compressed_embed = embed(compressed)
        og_compressed[passage] = {
            "compressed": compressed,
            "cosine": 1 - cosine(passage_embed, compressed_embed),
            "ratio": len(enc.encode(passage)) / len(enc.encode(compressed))
        }

dataset = []
for i, item in enumerate(ds['train']):
    entry = og_compressed[passages[i]]
    dataset.append({
        "passage": passages[i],
        "compressed_passage": entry["compressed"],
        "question": questions[i],
        "answer": item['answer'],
        "compression_ratio": entry["ratio"],
        "cosine_score": entry["cosine"],
        "requires_line": item['requires_line'],
        "id": item['id'],
    })

print(dataset[0])

with open("sat_compressed.json", "w") as f:
    json.dump(dataset, f, indent=2)
