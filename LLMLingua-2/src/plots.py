import json
import numpy as np
import matplotlib.pyplot as plt

# ── Load SAT reconstructed cosines ─────────────────────────────────────
with open("sat_reconstructed.json") as f:
    sat_recon = json.load(f)

sat_compressed_cosines    = [x[0] for x in sat_recon["original_compressed"]]
sat_reconstructed_cosines = [x[0] for x in sat_recon["original_reconstructed"]]

# ── Load SQuAD reconstructed cosines ───────────────────────────────────
with open("squad_reconstructed.json") as f:
    squad_recon = json.load(f)

squad_compressed_cosines    = [d["cosine_score"] for d in squad_recon]
squad_reconstructed_cosines = [d["reconstructed_cosine"] for d in squad_recon]

# ── Load QA results ────────────────────────────────────────────────────
with open("results_gpt_sat.json") as f:
    sat = json.load(f)

with open("results_gpt_squad.json") as f:
    squad = json.load(f)

# ── Helper ─────────────────────────────────────────────────────────────
def bin_by_ratio(data, n_bins=10):
    ratios     = np.array([d["compression_ratio"] for d in data])
    compressed = np.array([d["compressed_correct"] for d in data])
    original   = np.array([d["original_correct"] for d in data])

    bins = np.linspace(ratios.min(), ratios.max(), n_bins + 1)
    bin_centers, orig_acc, comp_acc = [], [], []

    for i in range(n_bins):
        mask = (ratios >= bins[i]) & (ratios < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            orig_acc.append(original[mask].mean())
            comp_acc.append(compressed[mask].mean())

    return bin_centers, orig_acc, comp_acc

# ── Plot 1: SAT vs SQuAD compressed accuracy by ratio ─────────────────
sat_bins,   _, sat_comp   = bin_by_ratio(sat)
squad_bins, _, squad_comp = bin_by_ratio(squad)

plt.figure(figsize=(8, 5))
plt.plot(sat_bins,   sat_comp,   marker='o', label='SAT English')
plt.plot(squad_bins, squad_comp, marker='s', label='SQuAD')
plt.xlabel("Compression Ratio")
plt.ylabel("Accuracy")
plt.title("Compression vs Accuracy: SAT English vs SQuAD")
plt.legend()
plt.tight_layout()
plt.savefig("plot_sat_vs_squad_accuracy.png")
print("Saved plot_sat_vs_squad_accuracy.png")

# ── Plot 2: SAT original vs compressed accuracy ────────────────────────
sat_bins, sat_orig, sat_comp = bin_by_ratio(sat)

plt.figure(figsize=(8, 5))
plt.plot(sat_bins, sat_orig, marker='o', linestyle='--', label='Original')
plt.plot(sat_bins, sat_comp, marker='s', label='Compressed')
plt.xlabel("Compression Ratio")
plt.ylabel("Accuracy")
plt.title("Original vs Compressed Accuracy by Ratio (SAT English)")
plt.legend()
plt.tight_layout()
plt.savefig("plot_sat_original_vs_compressed.png")
print("Saved plot_sat_original_vs_compressed.png")

# ── Plot 3: Cosine similarity SAT vs SQuAD ────────────────────────────
sat_c  = np.mean(sat_compressed_cosines)
sat_r  = np.mean(sat_reconstructed_cosines)
sq_c   = np.mean(squad_compressed_cosines)
sq_r   = np.mean(squad_reconstructed_cosines)

print(f"SAT  - Compressed: {sat_c:.4f}, Reconstructed: {sat_r:.4f}")
print(f"SQuAD - Compressed: {sq_c:.4f}, Reconstructed: {sq_r:.4f}")

x = np.arange(2)
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, [sat_c, sat_r], width, label='SAT English')
ax.bar(x + width/2, [sq_c,  sq_r],  width, label='SQuAD')
ax.set_xticks(x)
ax.set_xticklabels(['Original vs Compressed', 'Original vs Reconstructed'])
ax.set_ylabel("Cosine Similarity")
ax.set_title("Round-Trip Reconstruction Similarity: SAT vs SQuAD")
ax.legend()
plt.ylim(0.7, 1.0)
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