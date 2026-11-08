import time
import argparse
import os
from collections import deque
import numpy as np
import imageio
from game_env import DinoEnv
from agent import DQNAgent
from config import MODEL_SAVE_PATH

# GIF recording constants
GIF_FPS = 20
GIF_SECONDS = 30
GIF_PATH = "logs/best_run.gif"
GIF_FRAMES = GIF_FPS * GIF_SECONDS  # 600


def play(episodes=5, save_gif=False):
    env   = DinoEnv(headless=False)
    agent = DQNAgent()
    agent.load(MODEL_SAVE_PATH)  # epsilon automatically set to 0.0 for inference

    print(f"Loaded model from '{MODEL_SAVE_PATH}'. Running {episodes} episode(s)...\n")

    scores = []
    best_score = -1
    best_frames = None
    best_episode = 0
    current_frames = None
    interrupted = False

    try:
        for episode in range(1, episodes + 1):
            if save_gif:
                current_frames = deque(maxlen=GIF_FRAMES)
            
            state = env.reset()
            final_score = 0
            
            if save_gif:
                frame = env.get_raw_frame()
                if frame.mean() < 128:
                    frame = 255 - frame
                current_frames.append(frame)

            while True:
                action = agent.select_action(state)
                next_state, _, done, score = env.step(action)
                state = next_state
                final_score = score
                
                if save_gif:
                    frame = env.get_raw_frame()
                    if frame.mean() < 128:
                        frame = 255 - frame
                    current_frames.append(frame)

                if done:
                    break
            
            # Hold last frame for 10 extra frames after crash
            if save_gif and current_frames:
                last_frame = current_frames[-1]
                for _ in range(10):
                    current_frames.append(last_frame)
            
            # Update best episode
            if save_gif and final_score > best_score:
                best_score = final_score
                best_frames = list(current_frames)
                best_episode = episode

            scores.append(final_score)
            print(f"Episode {episode} | Score: {final_score:.0f}")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted by user (Ctrl+C).")
    
    finally:
        # Save GIF if recording was enabled
        if save_gif and (best_frames or current_frames):
            frames_to_save = best_frames if best_frames else list(current_frames)
            episode_label = best_episode if best_frames else "unfinished"
            score_label = best_score if best_frames else (scores[-1] if scores else 0)
            
            os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
            imageio.mimsave(GIF_PATH, frames_to_save, fps=GIF_FPS, loop=0)
            
            file_size_mb = os.path.getsize(GIF_PATH) / (1024 * 1024)
            clip_seconds = len(frames_to_save) / GIF_FPS
            
            print(f"\nGIF saved to: {GIF_PATH}")
            print(f"  Episode: {episode_label} | Score: {score_label:.0f}")
            print(f"  Frames: {len(frames_to_save)} | Duration: {clip_seconds:.1f}s | Size: {file_size_mb:.2f} MB")
            if best_frames:
                print(f"  Shows the last {GIF_SECONDS} seconds of the best episode out of {len(scores)} run.")
            else:
                print(f"  Shows the last {GIF_SECONDS} seconds of the unfinished current episode.")
        
        if not interrupted and scores:
            print(f"\nStatistics over {len(scores)} episodes:")
            print(f"  Mean:   {np.mean(scores):.0f}")
            print(f"  Std:    {np.std(scores):.0f}")
            print(f"  Median: {np.median(scores):.0f}")
            print(f"  Min:    {np.min(scores):.0f}")
            print(f"  Max:    {np.max(scores):.0f}")
        
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play DQN Dino agent")
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play (default: 5)"
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Record and save GIF of the best episode"
    )
    args = parser.parse_args()
    play(episodes=args.episodes, save_gif=args.save_gif)
