# What is LLMLingua-2?
LLMLingua-2 uses GPT-4 to create training pairs of original text and compressed text by prompting it to remove unnecessary tokens.
It then trains a small BERT-style encoder on this dataset. 
## Why a bidirectional encoder?
BERT looks at all tokens simultaneously 

## Causal vs. Masked LMs