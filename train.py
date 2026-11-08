import csv
import os
import argparse
import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from game_env import DinoEnv
from agent import DQNAgent
from config import (
    MAX_EPISODES, MAX_STEPS_PER_EPISODE, MAX_TOTAL_STEPS,
    TARGET_UPDATE_FREQ, MODEL_SAVE_PATH,
    OBSERVE_STEPS, EPSILON_END,
    SEED, EVAL_EVERY, EVAL_EPISODES, EVAL_MAX_STEPS,
)

os.makedirs("logs", exist_ok=True)
LOG_FILE  = "logs/training_log.csv"
EVAL_LOG_FILE = "logs/eval_log.csv"
PLOT_FILE = "logs/training_plot.png"


def save_plot(scores, avg_scores):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.4, color="steelblue", label="Score")
    plt.plot(avg_scores, color="navy", linewidth=2, label="Avg (100 ep)")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Score per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(avg_scores, color="navy", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Avg Score (100 ep)")
    plt.title("Learning Curve")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()


def evaluate(env, agent, episodes, max_steps):
    """
    Evaluate the agent for a fixed number of episodes.
    
    Returns list of episode scores. Does not modify agent state (epsilon is
    temporarily set to 0.0 and restored), does not store transitions, does not
    call learn() or decay_epsilon(), and does not affect total_steps.
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    eval_scores = []
    
    for _ in range(episodes):
        state = env.reset()
        episode_score = 0
        
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, _, done, score = env.step(action)
            state = next_state
            episode_score = score
            
            if done:
                break
        
        eval_scores.append(episode_score)
    
    agent.epsilon = original_epsilon
    return eval_scores


def train(save_all_improvements=False, eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES, eval_max_steps=EVAL_MAX_STEPS):
    # Seed for reproducibility
    # Note: the game's own randomness (obstacle spawning in the browser) is not seeded,
    # so runs are not exactly reproducible.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    env   = DinoEnv(headless=True)
    agent = DQNAgent()

    best_eval_score = 0
    scores     = []
    total_steps = 0  # cumulative steps across all episodes
    start_time = time.time()  # track elapsed time
    stop_reason = "max_episodes"  # default stop reason

    # Directory for improvement snapshots
    if save_all_improvements:
        os.makedirs("models", exist_ok=True)
        print("Saving every improvement to models/model_ep{episode}_score{score}.pth")

    # TensorBoard writer — logs go to logs/tensorboard/
    writer = SummaryWriter(log_dir="logs/tensorboard")

    # CSV log
    log_exists = os.path.exists(LOG_FILE)
    log_f  = open(LOG_FILE, "a", newline="")
    csv_writer = csv.writer(log_f)
    if not log_exists:
        csv_writer.writerow(["episode", "score", "avg_score_100", "epsilon", "steps", "total_steps"])
    
    # Eval CSV log
    eval_log_exists = os.path.exists(EVAL_LOG_FILE)
    eval_log_f = open(EVAL_LOG_FILE, "a", newline="")
    eval_csv_writer = csv.writer(eval_log_f)
    if not eval_log_exists:
        eval_csv_writer.writerow(["episode", "mean_score", "min_score", "max_score"])

    try:
        for episode in range(1, MAX_EPISODES + 1):
            state       = env.reset()
            final_score = 0
            episode_loss = []

            for step in range(MAX_STEPS_PER_EPISODE):
                action = agent.select_action(state)
                next_state, reward, done, score = env.step(action)

                agent.store(state, action, reward, next_state, done)
                total_steps += 1

                # Observation phase: fill the buffer before starting weight updates
                if total_steps > OBSERVE_STEPS:
                    loss = agent.learn()
                    if loss is not None:
                        episode_loss.append(loss)
                    agent.decay_epsilon()  # linear decay per step

                # Hard-copy target network every TARGET_UPDATE_FREQ steps
                if total_steps % TARGET_UPDATE_FREQ == 0:
                    agent.update_target_network()

                state       = next_state
                final_score = score

                if done:
                    break

            scores.append(final_score)
            avg_100 = sum(scores[-100:]) / min(len(scores), 100)

            # Log to CSV
            csv_writer.writerow([episode, f"{final_score:.0f}", f"{avg_100:.1f}",
                             f"{agent.epsilon:.6f}", step + 1, total_steps])
            log_f.flush()

            # Log to TensorBoard
            writer.add_scalar("Score/episode",         final_score,      episode)
            writer.add_scalar("Score/avg_100",          avg_100,          episode)
            writer.add_scalar("Agent/epsilon",          agent.epsilon,    episode)
            writer.add_scalar("Agent/steps_per_ep",     step + 1,         episode)
            writer.add_scalar("Agent/total_steps",      total_steps,      episode)
            writer.add_scalar("Replay/buffer_size",     len(agent.memory), episode)
            if episode_loss:
                avg_loss = sum(episode_loss) / len(episode_loss)
                writer.add_scalar("Agent/loss_avg", avg_loss, episode)
            
            # Check if we've reached the step limit
            reached_step_limit = total_steps >= MAX_TOTAL_STEPS
            is_last_episode = episode == MAX_EPISODES
            
            # Evaluation every eval_every episodes, at step limit, or after last episode
            should_evaluate = (episode % eval_every == 0) or reached_step_limit or is_last_episode
            
            if should_evaluate:
                elapsed = time.time() - start_time
                steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
                remaining_steps = max(0, MAX_TOTAL_STEPS - total_steps)
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                
                print(f"\n{'='*60}")
                print(f"Evaluating at episode {episode}...")
                print(f"Elapsed: {elapsed/3600:.2f}h | Total steps: {total_steps:,} | "
                      f"Steps/sec: {steps_per_sec:.1f} | ETA: {eta_seconds/3600:.2f}h")
                
                eval_scores = evaluate(env, agent, eval_episodes, eval_max_steps)
                eval_mean = sum(eval_scores) / len(eval_scores)
                eval_min = min(eval_scores)
                eval_max = max(eval_scores)
                
                print(f"Eval results: mean={eval_mean:.0f}, min={eval_min:.0f}, max={eval_max:.0f}")
                print(f"{'='*60}\n")
                
                # Log to TensorBoard
                writer.add_scalar("Eval/mean_score", eval_mean, episode)
                
                # Log to eval CSV
                eval_csv_writer.writerow([episode, f"{eval_mean:.0f}", f"{eval_min:.0f}", f"{eval_max:.0f}"])
                eval_log_f.flush()
                
                # Save best model based on evaluation score
                if eval_mean > best_eval_score:
                    best_eval_score = eval_mean
                    agent.save(MODEL_SAVE_PATH)
                    
                    if save_all_improvements:
                        snapshot_path = f"models/model_ep{episode:04d}_eval{eval_mean:.0f}.pth"
                        agent.save(snapshot_path)
                        print(f"  >> New best eval score! ep={episode} eval_mean={eval_mean:.0f} → {snapshot_path}")
                    else:
                        print(f"  >> New best eval score! Saved model at {eval_mean:.0f}\n")

            # Plot every 50 episodes
            if episode % 50 == 0:
                avg_scores = [
                    sum(scores[max(0, i - 99):i + 1]) / min(i + 1, 100)
                    for i in range(len(scores))
                ]
                save_plot(scores, avg_scores)

            # Phase label for console clarity
            if total_steps <= OBSERVE_STEPS:
                phase = "observe"
            elif agent.epsilon > EPSILON_END:
                phase = "explore"
            else:
                phase = "train"

            print(
                f"Episode {episode:4d} | "
                f"Score: {final_score:6.0f} | "
                f"Best Eval: {best_eval_score:6.0f} | "
                f"Avg(100): {avg_100:6.0f} | "
                f"Eps: {agent.epsilon:.5f} | "
                f"Steps: {step+1:5d} | "
                f"Total: {total_steps:7d} | "
                f"Phase: {phase}"
            )
            
            # Stop if we've reached the step limit
            if reached_step_limit:
                stop_reason = "max_total_steps"
                break
    
    finally:
        # Always clean up, even on Ctrl+C or exception
        log_f.close()
        eval_log_f.close()
        writer.close()
        
        # Final plot
        if len(scores) > 0:
            avg_scores = [
                sum(scores[max(0, i - 99):i + 1]) / min(i + 1, 100)
                for i in range(len(scores))
            ]
            save_plot(scores, avg_scores)
        
        # Save final model
        agent.save("last_model.pth")
        
        env.close()
        
        # Print completion message
        print(f"\nTraining stopped: {stop_reason}")
        print(f"Best eval score: {best_eval_score:.0f}")
        print(f"Training log saved to: {LOG_FILE}")
        print(f"Eval log saved to:     {EVAL_LOG_FILE}")
        print(f"Plot saved to:         {PLOT_FILE}")
        print(f"Final model saved to:  last_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Dino agent")
    parser.add_argument(
        "--save-all-improvements",
        action="store_true",
        help="Save a separate model snapshot for every new best score (into models/ folder)"
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=EVAL_EVERY,
        help=f"Evaluate every N episodes (default: {EVAL_EVERY})"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=EVAL_EPISODES,
        help=f"Number of evaluation episodes (default: {EVAL_EPISODES})"
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=EVAL_MAX_STEPS,
        help=f"Max steps per eval episode (default: {EVAL_MAX_STEPS})"
    )
    args = parser.parse_args()
    
    try:
        train(
            save_all_improvements=args.save_all_improvements,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            eval_max_steps=args.eval_max_steps
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C). Cleaning up...")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        raise
