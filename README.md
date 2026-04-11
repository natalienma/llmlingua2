This is based off of the paper: LLMLingua-2: Learn Compression Target via Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression

# What is LLMLingua-2?
LLMLingua-2 uses GPT-4 to create training pairs of original text and compressed text by prompting it to remove unnecessary tokens.
It then trains a small BERT-style encoder on this dataset. 

## What was wrong with LLMLingua-1?
It used information entropy from a causal LM (LLaMA) to measure uncertainty to score tokens and prune with respect to those scores. 

## Why a bidirectional encoder?
BERT looks at all tokens simultaneously while GPT looks unidirectionally (left only) 

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

# Sample 1 (Vanilla):
max_tokens = 100

temperature = 0.3

Original tokens: 100

Compressed tokens: 63

Compression ratio: 1.6x

Original:
```The effects of climate change on our water resources can have a big impact on our world and our lives. Patterns of where, when, and how much precipitation falls are changing as temperatures rise. Some areas are experiencing heavier rain events while others are having more droughts. Flooding is an increasing issue as our climate is changing. Compared to the beginning of the 20th century,  precipitation events are stronger, heavier, and more frequent across most of the United States. Drought is also becoming```

Compressed:
```Climate change effects on water resources impact world and lives. Precipitation patterns changing with rising temperatures. Some areas experiencing heavier rain, others more droughts. Flooding increasing with climate change. Compared to 20th century start, precipitation events stronger, heavier, more frequent across most United States. Drought increasing.```

# Quality Metrics:
## 1. Round Trip Reconstruction
After compressing the text, have GPT-4 reconstruct the original using the compressed text. Score the similarity using Cosine Similarity scoring.

Original vs Compressed Similarity: 
0.902 similarity 

Original vs Reconstructed Similarity: 
0.936 similarity

### GPT-4 reconstructed prompt is more semantically similar to the original than the compressed prompt is to the original. 

# Evaluation using SAT English Questions:
Through a SAT English dataset, passages were compressed with GPT-4. 
We then tested performance of a third-party open source model (llama3.2, not GPT or BERT) on questions with both original passages and compressed passages:

Original accuracy: 63.09%
Compressed accuracy: 54.36%

### Alignment 
**Alignment could not be achieved:**
1. Soft Scoring with GPT-4:
Prompt GPT-4 to score each word from 0.0-1.0, providing more nuance for pruning.
Problem: GPT-4 hallucinates due to lack of reasoning. This hallucination compounded over time and soon all scores were converging to 0.
```[0.9, 0.9, 0.9, 0.9, 0.1, 0.5, 0.1, 0.5, 0.1, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.1, 1.0, 1.0, 0.1, 0.5, 0.1, 0.1, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5, 1.0, 0.5, 0.1, 0.5, 0.5, 1.0, 0.1, 0.5, 0.1, 0.1, 0.5, 0.5, 0.1, 0.1, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]```

2. Binary scoring (from original paper):
Not working because GPT-4 paraphrased the passages in the dataset instead of just dropping tokens.
- Poetic writing: GPT paraphrases very aggressively to replace flowery language with semantically important words (and reorders alot)
- Informational writing: Expected GPT to paraphrase less, but it still discards almost the entire middle section. 

Note about SAT English compression: The average compression ratio is 4.5, very aggressive likely because the compression prompt was to “compress as aggressively as possible.” While this affects alignment heavily, it surprisingly didn’t really affect performance– the answers for the aggressively compressed prompts were still quite accurate, with only a 9% loss. This is interesting because SAT English is supposed to be semantically very dense. Often, questions will ask about the context in which a certain word is used. However, if that word is pruned, the answer will likely be incorrect.

3. Soft Scoring with Attention Weights:
For this sentence: “The Great Depression was a Severe Global Economic Crisis.” The scores are as follows:
[CLS]: 0.150
the: 0.038
great: 0.025
depression: 0.057
was: 0.053
a: 0.034
severe: 0.027
global: 0.034
economic: 0.032
crisis: 0.045
.: 0.130
[SEP]: 0.377
The problem is attention weights do not correspond to importance. 

