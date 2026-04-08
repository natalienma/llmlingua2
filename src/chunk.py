import tiktoken

# sliding window chunking
def chunk_text(input_text, max_tokens = 512):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(input_text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunks.append(chunk_tokens)
    return chunks