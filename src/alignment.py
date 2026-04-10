# score each token in original text from 0-1
# 0 = drop
# 1 = keep

# continuous scoring
import json
import re
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()

IMPORTANCE_PROMPT = """Rate each word's importance for SAT reading comprehension. 
- Function words like "the", "a", "is" = 0.1
- Content words like nouns, verbs, adjectives = 0.5-0.7  
- Key facts, names, dates, central concepts = 0.9-1.0

Return ONLY a JSON array of floats, one per word.

Passage: {passage}
Words: {words}"""

with open("sat_compressed.json") as f:
    data = json.load(f)

soft_scores = []

# method 1: GPT-4 rating (doesn't work, too much hallucination)
# for item in data:
#     words = item["passage"].split()
    
#     resp = client.chat.completions.create(
#         model="gpt-4",
#         messages=[{"role": "user", 
#                    "content": IMPORTANCE_PROMPT.format(passage=item["passage"], words=words
#                    )}],
#         max_tokens=4000,
#         temperature = 0
#     )
    
#     raw = resp.choices[0].message.content
#     print(raw)
#     match = re.search(r'\[.*\]', raw, re.DOTALL)
#     if match:
#         scores = json.loads(match.group())
#     else:
#         print(f"Failed to parse item {item['id']}: {raw[:200]}")
#         scores = None

#     soft_scores.append({"id": item["id"], "words": words, "scores": scores})

# with open("soft_scores.json", "w") as f:  
#     json.dump(soft_scores, f, indent=2)

# method 2: from existing data
