from dotenv import load_dotenv
load_dotenv()
import tiktoken
from openai import OpenAI
client = OpenAI()

COMPRESSION_PROMPT = "You are an excellent linguist. Compress the given text to short expressions, such that you can reconstruct it as close as possible to the original. Unlike usual text compression: Compress by removing words only — never add new words. Compress as aggressively as possible. Retain as much information as possible.Keep all named entities, numbers, dates. Output ONLY the compressed text, nothing else. Text to compress: {text}"

def compress_with_gpt(passage, max_tokens=2000):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(passage)
    if len(tokens) > max_tokens:
        passage = enc.decode(tokens[:max_tokens])
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", 
                   "content": COMPRESSION_PROMPT.format(text=passage)}],
        temperature=0
    )

    return resp.choices[0].message.content
     
