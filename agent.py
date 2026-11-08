import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    N_ACTIONS, FRAME_STACK, FRAME_WIDTH, FRAME_HEIGHT,
    LEARNING_RATE, GAMMA,
    EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS,
    BATCH_SIZE, REPLAY_BUFFER_SIZE,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ------------------------------------------------------------------ #
#  CNN adapted for the 20×80 frame shape                             #
# ------------------------------------------------------------------ #

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (4, 20, 80) — 4 stacked frames, 20px tall × 80px wide.
        # Using smaller kernels than the Atari DQN paper because the frame
        # height is only 20px; the original 8×8 kernel would blow up the dims.
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=4, stride=2),  # -> (32, 9, 39)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),           # -> (64, 4, 19)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # -> (64, 2, 17)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 17, 512),
            nn.ReLU(),
            nn.Linear(512, N_ACTIONS),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)


# ------------------------------------------------------------------ #
#  Agent                                                             #
# ------------------------------------------------------------------ #

class DQNAgent:
    def __init__(self):
        self.policy_net = DQN().to(DEVICE)
        self.target_net = DQN().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory    = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon   = EPSILON_START
        self.steps_done = 0  # total steps taken across all episodes

    def select_action(self, state):
        """Epsilon-greedy. state shape: (4, 20, 80)"""
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.policy_net(state_t).argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.tensor(np.array(states),      dtype=torch.float32).to(DEVICE)
        actions     = torch.tensor(actions,               dtype=torch.long).unsqueeze(1).to(DEVICE)
        rewards     = torch.tensor(rewards,               dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE)
        dones       = torch.tensor(dones,                 dtype=torch.float32).unsqueeze(1).to(DEVICE)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q   = rewards + GAMMA * max_next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        """Linear decay by one step. Call once per environment step."""
        self.steps_done += 1
        self.epsilon = max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END) * self.steps_done / EPSILON_DECAY_STEPS,
        )

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path, save_training_state=False):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            save_training_state: If True, saves complete training state for resumption.
                               If False (default), saves only network weights for inference.
        """
        if save_training_state:
            checkpoint = {
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
            }
            torch.save(checkpoint, path)
        else:
            # Backward compatible: save only weights for inference
            torch.save(self.policy_net.state_dict(), path)

    def load(self, path, load_training_state=False):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            load_training_state: If True, loads complete training state for resumption.
                               If False (default), loads only network weights for inference.
        """
        checkpoint = torch.load(path, map_location=DEVICE)
        
        # Handle both old (state_dict only) and new (full checkpoint) formats
        if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
            # New format: full checkpoint
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            
            if load_training_state:
                # Restore training state
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                if 'epsilon' in checkpoint:
                    self.epsilon = checkpoint['epsilon']
                if 'steps_done' in checkpoint:
                    self.steps_done = checkpoint['steps_done']
            else:
                # For inference, ensure epsilon is 0
                self.epsilon = 0.0
        else:
            # Old format: state_dict only (backward compatible)
            self.policy_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if not load_training_state:
                # For inference, ensure epsilon is 0
                self.epsilon = 0.0
