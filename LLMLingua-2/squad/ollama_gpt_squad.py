import ollama
import json 
import pandas as pd 

def answer_question(passage, question):
    resp = ollama.chat(model="llama3.2", messages=[
        {"role": "user", "content": f"{passage}\n\n{question}\nAnswer briefly."}
    ])
    return resp['message']['content']

with open("squad_compressed_gpt.json") as f:
    dataset = json.load(f)

results = []
correct_original, correct_compressed = 0, 0 

def check_answer(predicted, ground_truth):
    predicted = predicted.lower().strip()
    ground_truth = ground_truth.lower().strip()
    # strip common llama preambles
    for prefix in ["the answer is", "answer:", "i think", "based on"]:
        predicted = predicted.replace(prefix, "")
    return ground_truth in predicted

for item in dataset:
    original_answer = answer_question(item['passage'], item['question'])
    compressed_answer = answer_question(item['compressed_passage'], item['question'])

    results.append({
        **item,  # includes passage, compressed, question, answer, cosine, ratio
        "original_answer": original_answer,
        "compressed_answer": compressed_answer,
        "original_correct": check_answer(original_answer, item['answer']),
    "compressed_correct": check_answer(compressed_answer, item['answer']),
    })

with open("results_gpt_squad.json", "w") as f:
    json.dump(results, f, indent = 2)

df = pd.DataFrame(results)
print(f"Original accuracy: {df['original_correct'].mean():.2%}")
print(f"Compressed accuracy: {df['compressed_correct'].mean():.2%}")
