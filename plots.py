import matplotlib.pyplot as plt
import numpy as np

# ── 1. Compression vs Accuracy: GPT-4 vs BERT on SAT ──────────────────
fig, ax = plt.subplots(figsize=(8, 5))
compression_ratios = [1, 2, 3, 4, 5]
gpt4_accuracy =     [0.63, 0.60, 0.54, 0.50, 0.47]  # replace with real values
bert_accuracy =     [0.63, 0.61, 0.56, 0.52, 0.49]  # replace with real values

ax.plot(compression_ratios, gpt4_accuracy, marker='o', label='GPT-4')
ax.plot(compression_ratios, bert_accuracy, marker='s', label='BERT')
ax.axhline(0.63, linestyle='--', color='gray', label='No Compression')
ax.set_xlabel("Compression Ratio")
ax.set_ylabel("Accuracy")
ax.set_title("Compression vs Accuracy: GPT-4 vs BERT on SAT English")
ax.legend()
plt.tight_layout()
plt.savefig("plot1_sat_gpt4_vs_bert.png")

# ── 2. Compression vs Accuracy: GPT-4 on SAT vs SQuAD ─────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sat_ratios  = [1, 2, 3, 4, 5]
sat_acc     = [0.63, 0.60, 0.54, 0.50, 0.47]  # replace with real values
squad_ratios = [1, 1.5, 2, 2.5, 3]
squad_acc    = [0.70, 0.68, 0.65, 0.62, 0.60]  # replace with real values

ax.plot(sat_ratios, sat_acc, marker='o', label='SAT English')
ax.plot(squad_ratios, squad_acc, marker='s', label='SQuAD')
ax.set_xlabel("Compression Ratio")
ax.set_ylabel("Accuracy")
ax.set_title("Compression vs Accuracy: GPT-4 on SAT English vs SQuAD")
ax.legend()
plt.tight_layout()
plt.savefig("plot2_sat_vs_squad_accuracy.png")

# ── 3. Cosine Similarity: SAT vs SQuAD ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
categories = ['Original vs\nCompressed', 'Original vs\nReconstructed']
sat_scores   = [0.902, 0.936]        # replace with real values
squad_scores = [0.88, 0.92]          # replace with real values

x = np.arange(len(categories))
width = 0.35
ax.bar(x - width/2, sat_scores,   width, label='SAT English')
ax.bar(x + width/2, squad_scores, width, label='SQuAD')
ax.set_ylabel("Cosine Similarity")
ax.set_title("Round-Trip Reconstruction Similarity: SAT vs SQuAD")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0.8, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig("plot3_cosine_sat_vs_squad.png")