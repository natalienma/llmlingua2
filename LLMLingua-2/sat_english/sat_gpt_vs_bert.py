import json
import ollama

def answer_question(passage, question):
    resp = ollama.chat(model="llama3.2", messages=[
        {"role": "user", "content": f"{passage}\n\n{question}\nAnswer with just the letter."}
    ])
    return resp['message']['content'].strip()

with open("sat_bert_compressed.json") as f:
    data = json.load(f)

results = []
correct_orig, correct_gpt4, correct_bert = 0, 0, 0

for i, item in enumerate(data):
    print(f"Processing {i+1}/{len(data)}")
    
    orig_ans  = answer_question(item['passage'], item['question'])
    gpt4_ans  = answer_question(item['compressed_passage'], item['question'])
    bert_ans  = answer_question(item['bert_compressed'], item['question'])
    
    orig_correct  = orig_ans[0].upper()  == item['answer'].upper()
    gpt4_correct  = gpt4_ans[0].upper()  == item['answer'].upper()
    bert_correct  = bert_ans[0].upper()  == item['answer'].upper()
    
    correct_orig  += orig_correct
    correct_gpt4  += gpt4_correct
    correct_bert  += bert_correct
    
    results.append({
        **item,
        "orig_ans": orig_ans,
        "gpt4_ans": gpt4_ans,
        "bert_ans": bert_ans,
        "orig_correct": orig_correct,
        "gpt4_correct": gpt4_correct,
        "bert_correct": bert_correct,
    })

n = len(data)
print(f"\nOriginal accuracy:        {correct_orig/n:.2%}")
print(f"GPT-4 compressed accuracy: {correct_gpt4/n:.2%}")
print(f"BERT compressed accuracy:  {correct_bert/n:.2%}")

with open("final_results.json", "w") as f:
    json.dump(results, f, indent=2)

