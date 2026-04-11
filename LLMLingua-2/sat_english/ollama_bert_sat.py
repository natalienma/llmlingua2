import ollama
import json 
import pandas as pd 

def answer_question(passage, question):
    resp = ollama.chat(model="llama3.2", messages=[
        {"role": "user", "content": f"{passage}\n\n{question}\nAnswer with just the letter."}
    ])
    return resp['message']['content']

with open("sat_bert_compressed.json") as f:
    dataset = json.load(f)

results = []
correct_original, correct_compressed = 0, 0 

for item in dataset:
    original_answer = answer_question(item['passage'], item['question'])
    compressed_answer = answer_question(item['compressed_passage'], item['question'])

    results.append({
        **item,  # includes passage, compressed, question, answer, cosine, ratio
        "original_answer": original_answer,
        "compressed_answer": compressed_answer,
        "original_correct": original_answer[0].upper() == item['answer'].upper(),
        "compressed_correct": compressed_answer[0].upper() == item['answer'].upper(),
    })

with open("results_sat_bert.json", "w") as f:
    json.dump(results, f, indent = 2)

df = pd.DataFrame(results)
print(f"Original accuracy: {df['original_correct'].mean():.2%}")
print(f"Compressed accuracy: {df['compressed_correct'].mean():.2%}")
