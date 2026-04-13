"""
Agent play + video recorder for OBELIX.
Runs your trained PPO+LSTM agent and saves a video of the episode.

Usage:
  python agent_play.py --difficulty 3 --wall_obstacles --save_video
  python agent_play.py --difficulty 0 --save_video --video_out my_agent.mp4
"""

import argparse
import cv2
import numpy as np
import os

from obelix import OBELIX

# ── Import your agent ─────────────────────────────────────────────────────────
# Make sure agent.py and weights_tbptt.pth are in the same directory
import importlib.util
import sys

def load_agent(agent_path="agent.py"):
    spec = importlib.util.spec_from_file_location("agent", agent_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size",     type=int, default=500)
    parser.add_argument("--max_steps",      type=int, default=1000)
    parser.add_argument("--wall_obstacles", action="store_true")
    parser.add_argument("--difficulty",     type=int, default=3,
                        help="0=static, 2=blinking, 3=moving+blinking")
    parser.add_argument("--box_speed",      type=int, default=2)
    parser.add_argument("--seed",           type=int, default=50)
    parser.add_argument("--agent_path",     type=str, default="agent_ppo_lstm.py",
                        help="Path to your agent.py file")
    parser.add_argument("--save_video",     action="store_true",
                        help="Save episode as mp4 video")
    parser.add_argument("--video_out",      type=str, default="agent_episode.mp4",
                        help="Output video filename")
    parser.add_argument("--fps",            type=int, default=15,
                        help="Frames per second in output video")
    parser.add_argument("--show_window",    action="store_true",
                        help="Show live OpenCV window while recording")
    args = parser.parse_args()

    # ── Load agent ────────────────────────────────────────────────────────────
    print(f"Loading agent from: {args.agent_path}")
    policy = load_agent(args.agent_path)
    print("Agent loaded successfully.")

    # ── Create environment ────────────────────────────────────────────────────
    bot = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed,
    )

    obs = bot.reset(seed=args.seed)

    # ── Set up video writer ───────────────────────────────────────────────────
    video_writer = None
    frames = []

    if args.save_video:
        print(f"Will save video to: {args.video_out}")

    # ── Run episode ───────────────────────────────────────────────────────────
    episode_reward = 0
    rng = np.random.default_rng(args.seed)

    print(f"\nRunning episode | Difficulty: {args.difficulty} "
          f"| Wall: {args.wall_obstacles} | Seed: {args.seed}")
    print("-" * 50)

    for step in range(1, args.max_steps + 1):

        # Get action from agent
        action = policy(obs, rng)

        # Step environment
        obs, reward, done = bot.step(action, render=False)
        episode_reward += reward

        # Render current frame
        frame = bot.render_frame()

        # frame is an OpenCV image (BGR numpy array)
        if frame is not None:

            # ── Overlay text on frame ─────────────────────────────────────────
            display = frame.copy()

            cv2.putText(display,
                        f"Step: {step}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(display,
                        f"Reward: {episode_reward:.0f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(display,
                        f"Action: {action}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 1, cv2.LINE_AA)

            diff_label = {0: "Static", 2: "Blinking", 3: "Moving+Blink"}
            cv2.putText(display,
                        f"Level: {diff_label.get(args.difficulty, str(args.difficulty))}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 1, cv2.LINE_AA)

            # ── Collect frame for video ───────────────────────────────────────
            if args.save_video:
                frames.append(display.copy())

            # ── Show live window if requested ─────────────────────────────────
            if args.show_window:
                cv2.imshow("OBELIX Agent", display)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    print("Quit by user.")
                    break

        # ── Logging ───────────────────────────────────────────────────────────
        if step % 100 == 0:
            print(f"  Step {step:4d} | Action: {action:3s} "
                  f"| Reward: {episode_reward:.1f}")

        if done:
            print(f"\nEpisode finished at step {step}")
            print(f"Final reward: {episode_reward:.1f}")
            break

    else:
        print(f"\nEpisode hit max steps ({args.max_steps})")
        print(f"Final reward: {episode_reward:.1f}")

    # ── Save video ────────────────────────────────────────────────────────────
    if args.save_video and frames:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.video_out, fourcc, args.fps, (w, h))

        for f in frames:
            writer.write(f)
        writer.release()

        print(f"\nVideo saved: {args.video_out}")
        print(f"Frames: {len(frames)} | FPS: {args.fps} "
              f"| Duration: {len(frames)/args.fps:.1f}s")

        # ── Also try to convert to a more compatible format ───────────────────
        # If you have ffmpeg installed, this produces a better mp4
        ffmpeg_out = args.video_out.replace(".mp4", "_ffmpeg.mp4")
        ret = os.system(
            f"ffmpeg -y -i {args.video_out} "
            f"-vcodec libx264 -crf 23 {ffmpeg_out} 2>/dev/null"
        )
        if ret == 0:
            print(f"ffmpeg re-encoded video: {ffmpeg_out}")
        else:
            print("(ffmpeg not found — using raw mp4v output)")

    if args.show_window:
        cv2.destroyAllWindows()

    print("\nDone.")