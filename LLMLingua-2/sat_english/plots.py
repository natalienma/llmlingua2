import matplotlib.pyplot as plt
import pandas as pd 
import pickle
import json

with open("results_sat.json") as f:
    dataset = json.load(f)

# compression ratio vs cosine score (semantic similarity)
compression_ratios= [(float(item['compression_ratio'])) for item in dataset]
cosine_scores = [(float(item['cosine_score'])) for item in dataset]

plt.scatter(compression_ratios, cosine_scores)
plt.xlabel("Compression Ratio")
plt.ylabel("Cosine Similarity Score")
plt.title("Compression Ratio vs Cosine Similarity")
plt.show()

# compression ratio vs accuracy 
df = pd.DataFrame(dataset)
groups = df.groupby(pd.cut(df['compression_ratio'], bins = 20))
original_means = groups['original_correct'].mean()
compressed_means = groups['compressed_correct'].mean()

plt.plot( original_means.index.astype(str), original_means, label = "original mean scores")
plt.plot( compressed_means.index.astype(str), compressed_means, label = "compressed mean scores")
plt.xlabel("20 Groups of Compression Ratios")
plt.ylabel("Answer Accuracy")
plt.title("llama3.2 Performance on Original vs. Compressed SAT English Questions")
plt.ylim(0, 1)
plt.legend()
plt.show()

# Cosine Similarity across 20 Samples
with open("results.pkl", "rb") as f:
    original_compressed_scores = pickle.load(f)
    original_reconstructed_scores = pickle.load(f)

plt.plot(original_compressed_scores, label = "original and compressed similarity")
plt.plot(original_reconstructed_scores, label = "original and reconstructed similarity")
plt.xlabel("20 samples")
plt.ylabel("Cosine Similarity Score (0-1)")
plt.title("Semantic Similarity over 20 SAT English Samples")
plt.legend()
plt.show()