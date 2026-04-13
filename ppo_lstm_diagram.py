"""
Generate architecture diagram exactly matching the PPO+LSTM code.
Run: python generate_architecture.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def box(ax, cx, cy, w, h, title, subtitle=None,
        fc='#D6E4F0', ec='#2874A6', tsize=10, ssize=8):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.03",
                          facecolor=fc, edgecolor=ec,
                          linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    if subtitle:
        ax.text(cx, cy + h*0.13, title,
                ha='center', va='center',
                fontsize=tsize, fontweight='bold',
                color='#1A5276', zorder=4)
        ax.text(cx, cy - h*0.18, subtitle,
                ha='center', va='center',
                fontsize=ssize, color='#2E4057',
                style='italic', zorder=4)
    else:
        ax.text(cx, cy, title,
                ha='center', va='center',
                fontsize=tsize, fontweight='bold',
                color='#1A5276', zorder=4)


def arrow(ax, x1, y1, x2, y2, color='#555555', label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.6, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2 + 0.05, (y1+y2)/2
        ax.text(mx, my, label, fontsize=7.5,
                color=color, ha='left', va='center')


fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6)
ax.axis('off')

# ── Step labels (top) ────────────────────────────────────────────────────────
for x, label in [(1.1, '① Input'), (3.1, '② Feature\nExtraction'),
                 (5.6, '③ Memory\nCore'),
                 (8.2, '④ Actor\nHead'), (8.2, '④ Critic\nHead'),
                 (11.0, '⑤ Output')]:
    pass  # drawn inline below

# ── 1. Observation input ─────────────────────────────────────────────────────
box(ax, 1.1, 3.0, 1.6, 2.8,
    'Observation\n$s_t$',
    subtitle='18-bit binary\nvector',
    fc='#FDFEFE', ec='#7F8C8D', tsize=10, ssize=8.5)

# Bit breakdown inside
for i, (label, color) in enumerate([
        ('bits 0–7\nnear sonar', '#AED6F1'),
        ('bits 8–15\nfar sonar',  '#A9DFBF'),
        ('bit 16\nIR sensor',     '#FAD7A0'),
        ('bit 17\nstuck flag',    '#F1948A')]):
    bx = 1.1
    by = 3.85 - i * 0.62
    small = FancyBboxPatch((bx - 0.62, by - 0.22), 1.24, 0.44,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor='#AAA',
                           linewidth=0.8, zorder=4)
    ax.add_patch(small)
    ax.text(bx, by, label, ha='center', va='center',
            fontsize=6.5, zorder=5)

# ── 2. FC layer ───────────────────────────────────────────────────────────────
box(ax, 3.2, 3.0, 1.7, 1.2,
    'FC Layer',
    subtitle='Linear(18→64)\n+ Tanh',
    fc='#D6EAF8', ec='#2E86C1')

# Code comment
ax.text(3.2, 2.1, '# Feature extractor',
        ha='center', fontsize=7, color='#5D6D7E',
        family='monospace')

# ── 3. LSTM ───────────────────────────────────────────────────────────────────
box(ax, 5.6, 3.0, 1.8, 1.4,
    'LSTM',
    subtitle='input: 64\nhidden: 128',
    fc='#D5F5E3', ec='#1E8449', tsize=11)

# Hidden state loop arrow
ax.annotate('',
            xy=(5.6, 4.05), xytext=(5.6, 4.05),
            arrowprops=dict(arrowstyle='->',
                            color='#1E8449', lw=1.3,
                            connectionstyle='arc3,rad=-1.4'),
            zorder=2)
ax.text(6.72, 4.15, r'$(h_{t\!-\!1},\,c_{t\!-\!1})$',
        fontsize=8, color='#1E8449', ha='center')
ax.text(5.6, 1.9, '# Memory core\n# batch_first=True',
        ha='center', fontsize=7, color='#5D6D7E',
        family='monospace')

# ── 4a. Actor head ────────────────────────────────────────────────────────────
box(ax, 8.2, 4.3, 2.0, 1.0,
    'Actor Head',
    subtitle='Linear(128→5)\nCategorical logits',
    fc='#FDEBD0', ec='#D35400', ssize=8)

# ── 4b. Critic head ───────────────────────────────────────────────────────────
box(ax, 8.2, 1.7, 2.0, 1.0,
    'Critic Head',
    subtitle='Linear(128→1)\nscalar value',
    fc='#F9EBEA', ec='#C0392B', ssize=8)

# ── 5. Outputs ────────────────────────────────────────────────────────────────
box(ax, 11.0, 4.3, 1.6, 0.7,
    'Action $a_t$',
    subtitle='sample from\nCategorical($\\pi$)',
    fc='#FDFEFE', ec='#7F8C8D', tsize=9, ssize=7.5)

box(ax, 11.0, 1.7, 1.6, 0.7,
    'Value $V(s_t)$',
    subtitle='for GAE\nadvantage',
    fc='#FDFEFE', ec='#7F8C8D', tsize=9, ssize=7.5)

# ── Arrows ────────────────────────────────────────────────────────────────────
arrow(ax, 1.9,  3.0, 2.3,  3.0)        # obs → FC
arrow(ax, 4.05, 3.0, 4.7,  3.0)        # FC → LSTM
arrow(ax, 6.5,  3.5, 7.2,  4.3)        # LSTM → actor
arrow(ax, 6.5,  2.5, 7.2,  1.7)        # LSTM → critic
arrow(ax, 9.2,  4.3, 10.2, 4.3)        # actor → action
arrow(ax, 9.2,  1.7, 10.2, 1.7)        # critic → value

# Dimension labels on arrows
ax.text(2.1,  3.15, '18',  fontsize=7.5, color='#555', ha='center')
ax.text(4.38, 3.15, '64',  fontsize=7.5, color='#555', ha='center')
ax.text(6.95, 3.95, '128', fontsize=7.5, color='#555', ha='center')
ax.text(6.95, 2.05, '128', fontsize=7.5, color='#555', ha='center')

# ── PPO loss box (bottom) ─────────────────────────────────────────────────────
loss_text = (r'PPO loss:  $\mathcal{L} = '
             r'\underbrace{-\min(r_t\hat{A}_t,\ '
             r'\mathrm{clip}(r_t,1\pm\epsilon)\hat{A}_t)}'
             r'_{\mathcal{L}^{\mathrm{CLIP}}}\ +\ '
             r'0.5\underbrace{\|V_\theta - V_{\mathrm{targ}}\|^2}'
             r'_{\mathcal{L}^{\mathrm{VF}}}\ -\ '
             r'0.01\underbrace{H[\pi_\theta]}'
             r'_{\mathrm{entropy}}$')
ax.text(6.5, 0.55, loss_text,
        ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='#F8F9FA', edgecolor='#BDC3C7',
                  linewidth=1.2))

# ── Section labels (top strip) ────────────────────────────────────────────────
for cx, label in [(1.1,  'Input'),
                  (3.2,  'Feature\nextraction'),
                  (5.6,  'Memory\ncore'),
                  (8.2,  'Policy\nheads'),
                  (11.0, 'Output')]:
    ax.text(cx, 5.65, label, ha='center', va='center',
            fontsize=8.5, color='#717D7E',
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='#F2F3F4', edgecolor='#BDC3C7',
                      linewidth=0.8))

ax.set_title('Recurrent Actor-Critic architecture: PPO + LSTM\n'
             r'(hidden state $(h_t, c_t)$ carried step-to-step during rollout; '
             r'reset to $\mathbf{0}$ at episode start)',
             fontsize=12, pad=8)

plt.tight_layout()
plt.savefig('architecture.pdf')
plt.savefig('architecture.png')
print("Saved architecture.pdf and architecture.png")
plt.close()