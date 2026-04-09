import ollama
import json 

def answer_question(passage, question):
    resp = ollama.chat(model="llama3.2", messages=[
        {"role": "user", "content": f"{passage}\n\n{question}\nAnswer with just the letter."}
    ])
    return resp['message']['content']

with open("sat_compressed.json") as f:
    dataset = json.load(f)

correct_original, correct_compressed = 0, 0 

for item in dataset:
    original_answer = answer_question(item['passage'], item['question'])
    compressed_answer = answer_question(item['compressed_passage'], item['question'])

    correct_original += original_answer[0].upper().strip() == item['answer'].upper().strip()
    correct_compressed += compressed_answer[0].upper().strip() == item['answer'].upper().strip()
    
print(f"Original accuracy: {correct_original/len(dataset):.2%}")
print(f"Compressed accuracy: {correct_compressed/len(dataset):.2%}")