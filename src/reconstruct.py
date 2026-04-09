# Round Trip Reconstruction
# Prompt GPT-4 to reconstruct the original text from the compressed.
# Then score cosine similarity

import numpy as np
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
    reconstructed = resp.choices[0].message.content
    return reconstructed

def embed(text):
    resp = client.embeddings.create(
        model = "text-embedding-3-small",
        input = text
    )
    return np.array(resp.data[0].embedding)
