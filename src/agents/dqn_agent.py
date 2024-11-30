import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from src.agents.agent import Agent
import gymnasium as gym
from copy import deepcopy
from collections import deque


class DQNAgent(Agent):
    def __init__(self, env: gym.Env, model: nn.Module, hyperparameters: dict):
        self.env = env

        self.learning_rate = hyperparameters['learning_rate']
        self.batch_size = hyperparameters['batch_size']
        self.gamma = hyperparameters['gamma']
        self.epsilon = hyperparameters['epsilon_start']
        self.epsilon_end = hyperparameters['epsilon_end']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.target_update_frequency = hyperparameters['target_update_frequency']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = model.to(self.device)
        self.target_net = deepcopy(self.policy_net).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=hyperparameters['replay_buffer_size'])

    def select_action(self, state, mode='train'):
        if mode == 'train' and random.random() <= self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            q_values = self.policy_net(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def after_episode(self, episode):
        if episode % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = max(self.epsilon_end, self.epsilon * np.exp(-1.0 / self.epsilon_decay))
