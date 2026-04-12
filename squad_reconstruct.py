import json
import pickle
import numpy as np
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()

RECONSTRUCT_PROMPT = "The following text has been compressed by pruning unnecessary tokens. Reconstruct the original uncompressed text: {text}"

def reconstruct(compressed):
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", 
                   "content": RECONSTRUCT_PROMPT.format(text=compressed)}],
        temperature=0.3
    )
    return resp.choices[0].message.content

def embed(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp.data[0].embedding)

if __name__ == "__main__":
    with open("squad_compressed_gpt.json") as f:
        data = json.load(f)

    data = data[:20]

    for item in data:
        original_embed = embed(item["passage"])
        compressed_embed = embed(item["compressed_passage"])
        reconstructed_embed = embed(reconstruct(item["compressed_passage"]))

        item["cosine_score"] = 1 - cosine(original_embed, compressed_embed)
        item["reconstructed_cosine"] = 1 - cosine(original_embed, reconstructed_embed)

    with open("squad_reconstructed.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Done. Avg cosine (orig vs compressed): {np.mean([d['cosine_score'] for d in data]):.4f}")
    print(f"Done. Avg cosine (orig vs reconstructed): {np.mean([d['reconstructed_cosine'] for d in data]):.4f}")