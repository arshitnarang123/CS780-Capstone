"""
plot_utils.py — Shared logging and plotting utilities for OBELIX training.
Place in the same directory as your training scripts.

Usage in any training script:
    from plot_utils import TrainingLogger
    logger = TrainingLogger(run_name="ppo_lstm")
    logger.log(ep, ep_reward)          # call after each episode
    logger.save_plots()                # call at end of training
    logger.save_csv()                  # also saves raw data as CSV
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'savefig.dpi':    300,
    'savefig.bbox':   'tight',
})


class TrainingLogger:
    def __init__(self, run_name="run", log_dir="training_logs"):
        self.run_name   = run_name
        self.log_dir    = log_dir
        self.episodes   = []
        self.rewards    = []
        self.best_so_far = []
        self.wall_hits  = []    # optional
        self.grad_norms = []    # optional
        os.makedirs(log_dir, exist_ok=True)

    def log(self, episode, reward,
            wall_hits=None, grad_norm=None):
        self.episodes.append(episode)
        self.rewards.append(reward)
        best = reward if not self.best_so_far \
               else max(self.best_so_far[-1], reward)
        self.best_so_far.append(best)
        if wall_hits  is not None: self.wall_hits.append(wall_hits)
        if grad_norm  is not None: self.grad_norms.append(grad_norm)

    def _moving_avg(self, data, window=20):
        if len(data) < window:
            return np.array(data)
        return np.convolve(data,
                           np.ones(window)/window,
                           mode='valid')

    def save_csv(self):
        path = os.path.join(self.log_dir, f"{self.run_name}_log.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode', 'reward', 'best_so_far']
            if self.wall_hits:  header.append('wall_hits')
            if self.grad_norms: header.append('grad_norm')
            writer.writerow(header)
            for i, (ep, r, b) in enumerate(
                    zip(self.episodes, self.rewards, self.best_so_far)):
                row = [ep, r, b]
                if self.wall_hits:  row.append(self.wall_hits[i])
                if self.grad_norms: row.append(self.grad_norms[i])
                writer.writerow(row)
        print(f"[logger] CSV saved: {path}")

    def save_plots(self):
        eps     = np.array(self.episodes)
        rewards = np.array(self.rewards)
        best    = np.array(self.best_so_far)
        window  = min(20, len(rewards))
        ma      = self._moving_avg(rewards, window)
        ma_eps  = eps[window-1:] if len(eps) >= window else eps

        # ── Plot 1: Reward curve ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4.5))

        ax.scatter(eps, rewards, s=5, alpha=0.3,
                   color='steelblue', label='Episode reward', zorder=2)
        ax.plot(ma_eps, ma, color='steelblue', linewidth=2.0,
                label=f'Moving avg (w={window})', zorder=3)
        ax.plot(eps, best, color='crimson', linewidth=1.5,
                linestyle='--', label='Best so far', zorder=4)
        ax.axhline(0, color='gray', linewidth=0.7,
                   linestyle=':', alpha=0.6)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative reward')
        ax.set_title(f'Training reward — {self.run_name}')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.text(0.01, 0.02,
                'Negative spikes = wall collision episodes ($-200$ per hit)',
                transform=ax.transAxes, fontsize=8, color='gray')

        plt.tight_layout()
        p = os.path.join(self.log_dir, f"{self.run_name}_reward_curve.pdf")
        plt.savefig(p)
        plt.savefig(p.replace('.pdf', '.png'))
        print(f"[logger] Reward curve saved: {p}")
        plt.close()

        # ── Plot 2: Wall hits per episode (if logged) ─────────────────────────
        if self.wall_hits:
            fig, ax = plt.subplots(figsize=(9, 3.5))
            wh = np.array(self.wall_hits)
            ax.bar(eps, wh, color='salmon', alpha=0.7,
                   label='Wall hits', width=0.8)
            ma_wh = self._moving_avg(wh, window)
            ax.plot(ma_eps, ma_wh, color='darkred',
                    linewidth=1.8, label=f'Moving avg (w={window})')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Wall hits')
            ax.set_title(f'Wall collisions per episode — {self.run_name}')
            ax.legend(framealpha=0.9)
            plt.tight_layout()
            p = os.path.join(self.log_dir,
                             f"{self.run_name}_wall_hits.pdf")
            plt.savefig(p)
            plt.savefig(p.replace('.pdf', '.png'))
            print(f"[logger] Wall hits plot saved: {p}")
            plt.close()

        # ── Plot 3: Gradient norm (if logged) ────────────────────────────────
        if self.grad_norms:
            fig, ax = plt.subplots(figsize=(9, 3.5))
            gn = np.array(self.grad_norms)
            ax.plot(eps, gn, color='purple', linewidth=0.8,
                    alpha=0.6, label='Grad norm')
            ma_gn = self._moving_avg(gn, window)
            ax.plot(ma_eps, ma_gn, color='purple',
                    linewidth=2.0, label=f'Moving avg (w={window})')
            ax.axhline(0.5, color='red', linewidth=1.0,
                       linestyle='--', alpha=0.7,
                       label='Clip threshold (0.5)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Gradient norm (pre-clip)')
            ax.set_title(f'Gradient norm — {self.run_name}')
            ax.legend(framealpha=0.9)
            plt.tight_layout()
            p = os.path.join(self.log_dir,
                             f"{self.run_name}_grad_norm.pdf")
            plt.savefig(p)
            plt.savefig(p.replace('.pdf', '.png'))
            print(f"[logger] Grad norm plot saved: {p}")
            plt.close()

    @staticmethod
    def plot_comparison(log_dir="training_logs"):
        """
        Reads all *_log.csv files in log_dir and overlays reward curves.
        Call once after all training runs are done.
        """
        import glob
        files = sorted(glob.glob(os.path.join(log_dir, "*_log.csv")))
        if not files:
            print("[logger] No CSV files found for comparison plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        colors  = plt.cm.tab10.colors

        for i, fpath in enumerate(files):
            run_name = os.path.basename(fpath).replace('_log.csv', '')
            eps, rewards = [], []
            with open(fpath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    eps.append(int(row['episode']))
                    rewards.append(float(row['reward']))

            eps     = np.array(eps)
            rewards = np.array(rewards)
            window  = min(20, len(rewards))
            ma      = np.convolve(rewards,
                                  np.ones(window)/window,
                                  mode='valid')
            ma_eps  = eps[window-1:]

            color = colors[i % len(colors)]
            ax.plot(ma_eps, ma, linewidth=2.0,
                    color=color, label=run_name)
            ax.scatter(eps, rewards, s=3, alpha=0.15, color=color)

        ax.axhline(0, color='gray', linewidth=0.7,
                   linestyle=':', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative reward (moving avg)')
        ax.set_title('Training reward comparison across all variants')
        ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
        plt.tight_layout()
        p = os.path.join(log_dir, "comparison_all_runs.pdf")
        plt.savefig(p)
        plt.savefig(p.replace('.pdf', '.png'))
        print(f"[logger] Comparison plot saved: {p}")
        plt.close()