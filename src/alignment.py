# score each token in original text from 0-1
# 0 = drop
# 1 = keep

# improvement: continuous scoring
# method 1: GPT-4 rating
from openai import OpenAI
client = OpenAI()

def gpt_alignment(original_compressed_concat):
    IMPORTANCE_PROMPT = "Passage 1 is from the SAT English section. Passage 2 is a version of Passage 1 compressed by GPT-4. Your job is to rank the importance of each token in the original text from 0.0 to 1.0. "
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", 
                   "content": IMPORTANCE_PROMPT.format(text=original_compressed_concat)}],
    )
    return resp.choices[0].message.content

# method 2: from existing data
