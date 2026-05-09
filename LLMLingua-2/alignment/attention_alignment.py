# using attention weights as a proxy for token importance
# note: this doesn't really work because raw attention =/= importance
# attention tells you what tokens refer to each other, not what has the most impact
# keeping this here for documentation -- useful to understand why

from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# output_attentions=True returns attention matrix at every layer
model = AutoModel.from_pretrained(model_name, output_attentions=True)


def attention_scores(text):
    # tokenize and truncate at 512 tokens, BERT max = 512
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # no_grad because we're just doing inference
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of tensors (one per each layer)
    # tensor shape: (batch_size, num_heads, seq_len, seq_len)
    # stacking those tensors gives: (num_layers, batch_size, num_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions)  # [12, 1, 12, seq, seq]

    # average out layers (dim 0), batch (dim 1), and heads (dim 2)
    # left with a seq x seq matrix -- how much each token attends to each other token
    avg_attention = attentions.mean(dim=(0, 1, 2))  # [seq, seq]

    # collapse rows: for each token, how much attention did it receive from everyone else
    # this is the "importance" score -- but again, this is a flawed assumption
    token_importance = avg_attention.mean(dim=0)  # [seq]

    # pair each token string with its score
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return list(zip(tokens, token_importance.tolist()))

scores = attention_scores("The Great Depression was a severe global economic crisis.")

for token, score in scores:
    print(f"{token}: {score:.4f}")
