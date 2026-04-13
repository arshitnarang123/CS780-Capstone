"""
combine_plots.py — Combines multiple plots into a single figure for the report.
Place all your PDF/PNG plots in the same directory and run this script.

Run:
  python combine_plots.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

plt.rcParams.update({
    'font.family': 'serif',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── List your 10 plot files here in the order you want them ──────────────────
# Change filenames to match what you actually have in training_logs/
PLOTS = [
    ("training_logs/ppo_lstm_256_reward_curve.png",     "PPO+LSTM (h=256)"),
    ("training_logs/ppo_lstm_shaped_reward_curve.png",  "PPO+LSTM Shaped"),
    ("training_logs/ppo_gru_reward_curve.png",          "PPO GRU"),
    ("training_logs/ppo_lstm_tbptt_reward_curve.png",   "PPO LSTM TBPTT"),
    ("training_logs/ppo_gru_wall_hits.png",             "GRU Wall Hits"),
    ("training_logs/ppo_lstm_tbptt_wall_hits.png",      "TBPTT Wall Hits"),
    ("training_logs/ppo_gru_grad_norm.png",             "GRU Grad Norm"),
    ("training_logs/ppo_lstm_tbptt_grad_norm.png",      "TBPTT Grad Norm"),
    ("training_logs/comparison_all_runs.png",           "All Runs Comparison"),
]

# ── Layout: 2 columns x 5 rows ───────────────────────────────────────────────
NCOLS = 3
NROWS = 3

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(16, 12))
axes = axes.flatten()

for i, (fpath, title) in enumerate(PLOTS):
    ax = axes[i]
    if os.path.exists(fpath):
        img = mpimg.imread(fpath)
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    else:
        ax.text(0.5, 0.5, f"File not found:\n{fpath}",
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=9, color='red')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
    ax.axis('off')

plt.suptitle("Training Results",
             fontsize=14, fontweight='bold', y=1.005)

plt.tight_layout(h_pad=2.0, w_pad=1.5)
plt.savefig("combined_plots.pdf", bbox_inches='tight')
plt.savefig("combined_plots.png", bbox_inches='tight')
print("Saved: combined_plots.pdf and combined_plots.png")
plt.close()