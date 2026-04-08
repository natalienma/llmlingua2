from openai import OpenAI
client = OpenAI()

COMPRESSION_PROMPT = """You are an excellent linguist. Compress the given 
text to short expressions, such that you can reconstruct it as close as 
possible to the original. Unlike usual text compression:
1. Compress by removing words only — never add new words
2. Compress as aggressively as possible  
3. Retain as much information as possible
4. Keep all named entities, numbers, dates
5. Output ONLY the compressed text, nothing else

Text to compress:
{text}"""

def compress_with_gpt(input_text):
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", 
                   "content": COMPRESSION_PROMPT.format(text=input_text)}],
        temperature=0.3
    )
    return resp.choices[0].message.content

process.env()