# Vanilla Policy Gradient with reward-to-go and value function baseline

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.agents.agent import Agent
import gymnasium as gym
from src.utils.network_builder import build_network


class VPGAgent(Agent):
    def __init__(self, env: gym.Env, hyperparameters: dict):
        self.env = env

        self.learning_rate = hyperparameters['learning_rate']
        self.gamma = hyperparameters['gamma']
        self.batch_size = hyperparameters['batch_size']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        architecture_config = hyperparameters['network_architecture']
        self.policy_net = build_network(
            env.observation_space.shape[0],
            env.action_space.n,
            architecture_config['policy_network'],
        ).to(self.device)
        self.value_net = build_network(
            env.observation_space.shape[0],
            1,
            architecture_config['value_network']
        ).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.value_criterion = nn.MSELoss()

        self.trajectory = [[]]

    def select_action(self, state, mode='train'):
        with torch.no_grad():
            logits = self.policy_net(torch.tensor(state, dtype=torch.float32))
            action = torch.distributions.Categorical(logits=logits).sample().item()
            return action

    def learn(self, state, action, reward, next_state, done):
        self.trajectory[-1].append((state, action, reward, next_state, done))

    def _get_rewards_to_go(self, rewards):
        result = []
        sum_r = 0
        for r in reversed(rewards):
            sum_r = r + self.gamma * sum_r
            result.append(sum_r)
        return list(reversed(result))

    def after_episode(self, episode):
        if len(self.trajectory) < self.batch_size:
            self.trajectory.append([])
            return

        states = []
        actions = []
        rewards_to_go = []

        for trajectory in self.trajectory:
            t_states, t_actions, t_rewards, t_next_states, t_dones = zip(*trajectory)
            states.extend(t_states)
            actions.extend(t_actions)

            rewards_to_go.extend(self._get_rewards_to_go(t_rewards))

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        value_predictions = self.value_net(states).squeeze()

        rewards_to_go = np.array(rewards_to_go)
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32, requires_grad=False).to(self.device)
        rewards_to_go -= value_predictions.detach()

        distributions = torch.distributions.Categorical(logits=self.policy_net(states))
        log_probs = distributions.log_prob(actions)
        policy_loss = -(log_probs * rewards_to_go).mean()

        value_loss = self.value_criterion(value_predictions, rewards_to_go)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.trajectory = [[]]
