import matplotlib.pyplot as plt
import pandas as pd 
import json

with open("results_sat.json") as f:
    dataset = json.load(f)

# compression ratio vs cosine score (semantic similarity)
compression_ratios= [(float(item['compression_ratio'])) for item in dataset]
cosine_scores = [(float(item['cosine_score'])) for item in dataset]

# plt.scatter(compression_ratios, cosine_scores)
# plt.xlabel("Compression Ratio")
# plt.ylabel("Cosine Similarity Score")
# plt.title("Compression Ratio vs Cosine Similarity")
# plt.show()

# compression ratio vs accuracy 
df = pd.DataFrame(dataset)
groups = df.groupby(pd.cut(df['compression_ratio'], bins = 5))
original_means = groups['original_correct'].mean()
compressed_means = groups['compressed_correct'].mean()

plt.plot( original_means.index.astype(str), original_means, label = "original mean scores")
plt.plot( compressed_means.index.astype(str), compressed_means, label = "compressed mean scores")
plt.xlabel("5 Groups of Compression Ratios")
plt.ylabel("Answer Accuracy")
plt.title("Ollama Performance on Original vs. Compressed SAT English Questions")
plt.ylim(0, 1)
plt.legend()
plt.show()