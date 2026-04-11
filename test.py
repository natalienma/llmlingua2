import json
with open("sat_bert_compressed.json") as f:
    data = json.load(f)
print(data[0].keys())
