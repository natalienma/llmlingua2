from datasets import load_dataset
from collections import defaultdict
from compress import compress_with_gpt

ds = load_dataset("emozilla/sat-reading")

def separate(ds): #split all passages and questions
    passages = []
    questions = []
    # somehow create a new category so there are 5: question, passage, answer, requires_line, id
    for item in ds['train']:
        parts = item['text'].split("Question")
        passages.append(parts[0])
        questions.append(parts[1])
    return passages, questions

passages, questions = separate(ds)

# create dictionary to get rid of repeating passages
# final dict will have 1 of each passage, 1 associated compressed
og_compressed = {}
for passage in passages:
    if passage not in og_compressed:
        og_compressed[passage] = compress_with_gpt(passage)

dataset = []
for i, item in enumerate(ds['train']):
    dataset.append({
        "passage": passages[i],
        "question": questions[i],
        "compressed_passage" : og_compressed[passages[i]],
        "answer": item['answer'],
        "requires_line": item['requires_line'],
        "id": item['id'],
    })

print(dataset[0])