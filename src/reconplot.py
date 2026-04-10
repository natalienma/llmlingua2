import matplotlib.pyplot as plt
import pickle

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