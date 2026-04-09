import matplotlib.pyplot as plt
import json

with open("results_sat.json") as f:
    dataset = json.load(f)

compression_ratios = []
cosine_scores = []
for item in dataset:
    compression_ratios.append(float(item['compression_ratio']))
    cosine_scores.append(float(item['cosine_score']))

#plot
plt.scatter(compression_ratios, cosine_scores)
plt.xlabel("Compression Ratio")
plt.ylabel("Cosine Similarity Score")
plt.title("Compression Ratio vs Cosine Similarity")
plt.show()