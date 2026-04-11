import spacy
import json
from datasets import load_dataset

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def tokenize(text):
    doc = nlp(text)
    return [word.lemma_ for word in doc if word.lemma_ != ","]

def align_labels(original, compressed):
    orig_tokens = tokenize(original)
    comp_tokens = tokenize(compressed)
    
    labels = [0] * len(orig_tokens)
    prev_idx = 0
    window = 20
    
    for comp_token in comp_tokens:
        matched = False
        for i in range(window):
            fwd = min(prev_idx + i, len(orig_tokens) - 1)
            if orig_tokens[fwd].lower() == comp_token.lower() and not labels[fwd]:
                labels[fwd] = 1
                prev_idx = fwd
                matched = True
                break
            bwd = max(prev_idx - i, 0)
            if orig_tokens[bwd].lower() == comp_token.lower() and not labels[bwd]:
                labels[bwd] = 1
                prev_idx = bwd
                matched = True
                break
        
        if not matched:
            for i in range(prev_idx, len(orig_tokens)):
                if orig_tokens[i].lower() == comp_token.lower() and not labels[i]:
                    labels[i] = 1
                    prev_idx = i
                    break
    
    return orig_tokens, labels

ds = load_dataset("microsoft/MeetingBank-LLMCompressed", split="train")
print(f"Dataset size: {len(ds)}")

results = []
total = 0
too_low_comp = 0
too_high_comp = 0
low_matching = 0

for sample in ds:
    for origin, comp in zip(sample['prompt_list'], sample['compressed_prompt_list']):
        orig_tokens, labels = align_labels(origin, comp)
        total += 1
        matching_rate = sum(labels) / len(labels) if labels else 0
        comp_rate = len(orig_tokens) / max(len(tokenize(comp)), 1)
        
        print(f"comp_rate: {comp_rate:.2f}, matching_rate: {matching_rate:.2f}")

        if comp_rate < 1.5:
            too_low_comp += 1
            continue
        if comp_rate > 10:
            too_high_comp += 1
            continue
        if matching_rate < 0.1:
            low_matching += 1
            continue
            
        results.append({
            "origin_tokens": orig_tokens,
            "labels": labels,
            "matching_rate": matching_rate,
            "comp_rate": comp_rate,
        })

print(f"Filtered - comp too low (<1.5): {too_low_comp}")
print(f"Filtered - comp too high (>10): {too_high_comp}")
print(f"Filtered - low matching (<0.1): {low_matching}")
print(f"Passed: {len(results)}")

with open("meetingbank_labeled.json", "w") as f:
    json.dump(results, f, indent=2)
