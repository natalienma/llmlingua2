import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ── Load SQuAD data ──────────────────────────────────────────────────────
with open("squad_reconstructed.json") as f:
    squad = json.load(f)

# ── Load SAT data ────────────────────────────────────────────────────
with open("sat_reconstructed.json") as f:
    sat = json.load(f)

with open("results.pkl", "rb") as f:
    original_compressed_scores = pickle.load(f)
    original_reconstructed_scores = pickle.load(f)

for i, item in enumerate(squad[:20]):
    item["cosine_score"] = original_compressed_scores[i][0]
    item["reconstructed_cosine"] = original_reconstructed_scores[i][0]

# ── Load SQuAD QA results ──────────────────────────────────────────────
with open("results_gpt_squad.json") as f:
    squad_results = json.load(f)

# ── Helpers ────────────────────────────────────────────────────────────
def bin_by_ratio(data, n_bins=10):
    ratios = np.array([d["compression_ratio"] for d in data])
    original = np.array([d["original_correct"] for d in data])
    compressed = np.array([d["compressed_correct"] for d in data])

    bins = np.linspace(ratios.min(), ratios.max(), n_bins + 1)
    bin_centers, orig_acc, comp_acc = [], [], []

    for i in range(n_bins):
        mask = (ratios >= bins[i]) & (ratios < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            orig_acc.append(original[mask].mean())
            comp_acc.append(compressed[mask].mean())

    return bin_centers, orig_acc, comp_acc

# # ── Plot 1: SAT vs SQuAD compressed accuracy by ratio ─────────────────
# sat_bins, _, sat_comp     = bin_by_ratio(sat)
# squad_bins, _, squad_comp = bin_by_ratio(squad_results)

# plt.figure(figsize=(8, 5))
# plt.plot(sat_bins, sat_comp, marker='o', label='SAT English')
# plt.plot(squad_bins, squad_comp, marker='s', label='SQuAD')
# plt.xlabel("Compression Ratio")
# plt.ylabel("Accuracy")
# plt.title("Compression vs Accuracy: SAT English vs SQuAD")
# plt.legend()
# plt.tight_layout()
# plt.savefig("plot_sat_vs_squad.png")
# print("Saved plot_sat_vs_squad.png")

# # ── Plot 2: SAT original vs compressed accuracy by ratio ───────────────
# sat_bins, sat_orig, sat_comp = bin_by_ratio(sat)

# plt.figure(figsize=(8, 5))
# plt.plot(sat_bins, sat_orig, marker='o', linestyle='--', label='Original')
# plt.plot(sat_bins, sat_comp, marker='s', label='Compressed')
# plt.xlabel("Compression Ratio")
# plt.ylabel("Accuracy")
# plt.title("Original vs Compressed Accuracy by Ratio (SAT English)")
# plt.legend()
# plt.tight_layout()
# plt.savefig("plot_sat_original_vs_compressed.png")
# print("Saved plot_sat_original_vs_compressed.png")

# ── Plot 3: Cosine similarity SAT vs SQuAD ────────────────────────────
sat_c  = np.mean([d["cosine_score"] for d in sat])
sat_r  = np.mean([d["reconstructed_cosine"] for d in sat])
sq_c   = np.mean([d["cosine_score"] for d in squad[:20]])
sq_r   = np.mean([d["reconstructed_cosine"] for d in squad[:20]])

x = np.arange(2)
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, [sat_c, sat_r], width, label='SAT English')
ax.bar(x + width/2, [sq_c,  sq_r],  width, label='SQuAD')
ax.set_xticks(x)
ax.set_xticklabels(['Original vs Compressed', 'Original vs Reconstructed'])
ax.set_ylim(0.8, 1.0)
ax.set_ylabel("Cosine Similarity")
ax.set_title("Round-Trip Reconstruction Similarity: SAT vs SQuAD")
ax.legend()
plt.tight_layout()
plt.savefig("plot_cosine_comparison.png")
print("Saved plot_cosine_comparison.png")

# ── Plot 4: Compression ratio distribution by domain ──────────────────
sat_ratios   = [d["compression_ratio"] for d in sat]
squad_ratios = [d["compression_ratio"] for d in squad]

plt.figure(figsize=(8, 5))
plt.hist(sat_ratios,   bins=15, alpha=0.6, label='SAT English')
plt.hist(squad_ratios, bins=15, alpha=0.6, label='SQuAD')
plt.xlabel("Compression Ratio")
plt.ylabel("Count")
plt.title("Compression Ratio Distribution by Domain")
plt.legend()
plt.tight_layout()
plt.savefig("plot_ratio_distribution.png")
print("Saved plot_ratio_distribution.png")