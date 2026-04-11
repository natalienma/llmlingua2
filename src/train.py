from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import json
import numpy as np
from sklearn.metrics import f1_score

# Load labeled data
with open("meetingbank_labeled.json") as f:
    data = json.load(f)

# Convert to HuggingFace dataset
hf_data = Dataset.from_list(data)
hf_data = hf_data.train_test_split(test_size=0.1)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["origin_tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        token_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                token_labels.append(-100)
            elif word_id != prev_word_id:
                token_labels.append(labels[word_id])
            else:
                token_labels.append(-100)
            prev_word_id = word_id
        all_labels.append(token_labels)
    tokenized["labels"] = all_labels
    return tokenized

tokenized_data = hf_data.map(tokenize_and_align, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    true, pred = [], []
    for p_seq, l_seq in zip(predictions, labels):
        for p, l in zip(p_seq, l_seq):
            if l != -100:
                true.append(l)
                pred.append(p)
    return {"f1": f1_score(true, pred, average="binary")}

args = TrainingArguments(
    output_dir="./bert-compressor",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./bert-compressor-final")
