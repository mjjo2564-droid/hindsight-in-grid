import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from gridworld import GridWorld
from Qnetwork import QNetwork


EPISODES = 3000
MAX_STEPS = 128

BATCH_SIZE = 64
BUFFER_SIZE = 100000

GAMMA = 0.98
LEARNING_RATE = 0.001

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

TARGET_UPDATE = 200
SEED = 0


def move(state, action, x_size, y_size):
    x, y = state

    if action == 0:
        y = min(y + 1, y_size - 1)
    elif action == 1:
        y = max(y - 1, 0)
    elif action == 2:
        x = max(x - 1, 0)
    elif action == 3:
        x = min(x + 1, x_size - 1)

    return x, y


def make_input(state, goal, x_size, y_size):
    x_scale = max(x_size - 1, 1)
    y_scale = max(y_size - 1, 1)

    return np.array(
        [
            state[0] / x_scale,
            state[1] / y_scale,
            goal[0] / x_scale,
            goal[1] / y_scale,
        ],
        dtype=np.float32,
    )


def learn(online, target, optimizer, replay, x_size, y_size):
    if len(replay) < BATCH_SIZE:
        return

    batch = random.sample(replay, BATCH_SIZE)
    states, actions, rewards, next_states, goals, dones = zip(*batch)

    state_inputs = torch.tensor(
        np.stack(
            [
                make_input(state, goal, x_size, y_size)
                for state, goal in zip(states, goals)
            ]
        ),
        dtype=torch.float32,
    )

    next_inputs = torch.tensor(
        np.stack(
            [
                make_input(next_state, goal, x_size, y_size)
                for next_state, goal in zip(next_states, goals)
            ]
        ),
        dtype=torch.float32,
    )

    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    current_q = online(state_inputs).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q = target(next_inputs).max(dim=1).values
        target_q = rewards + GAMMA * (1.0 - dones) * next_q

    loss = F.mse_loss(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_graph(history):
    output_dir = "results/128"
    os.makedirs(output_dir, exist_ok=True)


    successes = np.array(history, dtype=np.float32)
    episodes = np.arange(1, len(successes) + 1)

    window = 20
    moving_average = np.convolve(successes, np.ones(window) / window, mode="valid")
    moving_episodes = np.arange(window, len(successes) + 1)

    plt.figure(figsize=(10, 5))
    plt.scatter(episodes, successes, s=6, alpha=0.25, label="Episode result")
    plt.plot(moving_episodes, moving_average, color="red", linewidth=2, label="20-episode success rate")
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.title("DQN without HER")
    plt.xlim(1, EPISODES)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0, 1], ["Failure", "Success"])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dqn_noher.png"))
    plt.close()


def train():
    random.seed(SEED)
    torch.manual_seed(SEED)

    env = GridWorld(seed=SEED)
    x_size = env.x_size
    y_size = env.y_size

    online = QNetwork()
    target = QNetwork()
    target.load_state_dict(online.state_dict())

    optimizer = optim.Adam(online.parameters(), lr=LEARNING_RATE)
    replay = deque(maxlen=BUFFER_SIZE)

    epsilon = EPSILON_START
    total_steps = 0
    history = []

    for episode in range(1, EPISODES + 1):
        env.reset()

        state = env.start
        goal = env.goal
        success = False

        for step in range(1, MAX_STEPS + 1):
            if random.random() < epsilon:
                action = random.randrange(4)
            else:
                network_input = torch.tensor(make_input(state, goal, x_size, y_size), dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    action = online(network_input).argmax(dim=1).item()

            next_state = move(state, action, x_size, y_size)
            success = next_state == goal
            reward = 0.0 if success else -1.0

            replay.append((state, action, reward, next_state, goal, success))

            state = next_state
            total_steps += 1
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            learn(online, target, optimizer, replay, x_size, y_size)

            if total_steps % TARGET_UPDATE == 0:
                target.load_state_dict(online.state_dict())

            if success:
                break

        history.append(int(success))

        print(f"Episode: {episode:3d} | Goal: {goal} | Success: {success} | Steps: {step}")

    save_graph(history)


if __name__ == "__main__":
    train()