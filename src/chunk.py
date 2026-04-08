import tiktoken

# sliding window chunking
def chunk_text(text, max_tokens = 100):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunks.append(chunk_tokens)
    return chunks