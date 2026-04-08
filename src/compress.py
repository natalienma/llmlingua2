from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

COMPRESSION_PROMPT = "Compress by removing words only, never add new words. Output ONLY the compressed text.\n\nText:\n{text}"

def compress_with_gpt(chunk0):
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", 
                   "content": COMPRESSION_PROMPT.format(text=chunk0)}],
        temperature=0.3
    )
    return resp.choices[0].message.content
