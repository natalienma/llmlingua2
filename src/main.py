# from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

ds = load_dataset("microsoft/MeetingBank-LLMCompressed")
print(ds)

sample = ds['train'][0]
print("Original:", sample['prompt'][:300])
print("Compressed:", sample['compressed_prompt'][:300])
