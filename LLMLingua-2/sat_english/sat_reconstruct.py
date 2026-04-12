# Round Trip Reconstruction
# Prompt GPT-4 to reconstruct the original text from the compressed.
# Then score cosine similarity

import numpy as np
import json
import pickle
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
        temperature=0
    )
    reconstructed = resp.choices[0].message.content
    return reconstructed

def embed(text):
    resp = client.embeddings.create(
        model = "text-embedding-3-small",
        input = text
    )
    return np.array(resp.data[0].embedding)

if __name__ == "__main__":
    with open("sat_compressed.json") as f:
        data = json.load(f)

    original = [i["passage"] for i in data]
    compressed = [i["compressed_passage"] for i in data]

    original_compressed_scores = []
    original_reconstructed_scores = []
    for i in range(20):
        original_embed = embed(original[i])
        compressed_embed = embed(compressed[i])
        reconstructed_embed = embed(reconstruct(compressed[i]))
        
        original_compressed_scores.append([1- cosine(original_embed, compressed_embed)])
        original_reconstructed_scores.append([1- cosine(original_embed, reconstructed_embed)])

    with open("sat_reconstructed_cosines.pkl", "wb") as f:
        pickle.dump(original_compressed_scores, f)
        pickle.dump(original_reconstructed_scores, f)

