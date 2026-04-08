from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

COMPRESSION_PROMPT = "You are an excellent linguist. Compress the given text to short expressions, such that you can reconstruct it as close as possible to the original. Unlike usual text compression: Compress by removing words only — never add new words. Compress as aggressively as possible. Retain as much information as possible.Keep all named entities, numbers, dates. Output ONLY the compressed text, nothing else. Text to compress: {text}"

def compress_with_gpt(chunk0):
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", 
                   "content": COMPRESSION_PROMPT.format(text=chunk0)}],
        temperature=0.3
    )
    return resp.choices[0].message.content
