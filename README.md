This is based off of the paper: LLMLingua-2: Learn Compression Target via Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression

# What is LLMLingua-2?
LLMLingua-2 uses GPT-4 to create training pairs of original text and compressed text by prompting it to remove unnecessary tokens.
It then trains a small BERT-style encoder on this dataset. 

## What was wrong with LLMLingua-1?
It used information entropy from a causal LM (LLaMA) to measure uncertainty to score tokens and prune with respect to those scores. 

## Why a bidirectional encoder?
BERT looks at all tokens simultaneously 

## Causal vs. Masked LMs

# Prompt used in the paper:
COMPRESSION_PROMPT = """You are an excellent linguist. Compress the given 
text to short expressions, such that you can reconstruct it as close as 
possible to the original. Unlike usual text compression:
1. Compress by removing words only — never add new words
2. Compress as aggressively as possible  
3. Retain as much information as possible
4. Keep all named entities, numbers, dates
5. Output ONLY the compressed text, nothing else

Text to compress:
{text}"""