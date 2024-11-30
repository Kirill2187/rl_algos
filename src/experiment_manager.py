import os
from src.agents import DQNAgent
from src.agents import VPGAgent
from src.utils.logger import Logger
from src.utils.seeding import set_seeds
from src.utils.visualization import show_episode
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch

SEED = 42
set_seeds(SEED)


class ExperimentManager:
    def __init__(self, config):
        self.video_dir = config['logging'].get('video_dir')
        if self.video_dir is not None:
            os.makedirs(self.video_dir, exist_ok=True)
        self.show_episode = config['logging'].get('show_episode', False)
        self.save_video_frequency = config['logging'].get('save_video_frequency', 0)

        self.config = config
        self.env = RecordVideo(
            gym.make(
                config['environment_name'],
                render_mode='rgb_array' if self.save_video_frequency > 0 else None,
                max_episode_steps=config['training'].get('max_steps_per_episode')
            ),
            video_folder=self.video_dir,
            episode_trigger=lambda e: e % self.save_video_frequency == 0 if self.save_video_frequency > 0 else False
        )
        self.env.action_space.seed(SEED)
        self.logger = Logger(config)
        self.agent = self._initialize_agent()
        self.total_steps = 0

    def _initialize_agent(self):
        agent_type = self.config['agent']['type']
        if agent_type == 'DQN':
            return DQNAgent(self.env, self.config['agent']['hyperparameters'])
        elif agent_type == 'VPG':
            return VPGAgent(self.env, self.config['agent']['hyperparameters'])
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

    def run(self):
        num_episodes = self.config['training']['num_episodes']
        running_total_reward = 0

        for episode in range(num_episodes):
            state, _ = self.env.reset(seed=SEED)
            total_reward = 0
            while True:
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.agent.learn(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                self.total_steps += 1
                if terminated or truncated:
                    break

            self.agent.after_episode(episode)

            running_total_reward = 0.95 * running_total_reward + 0.05 * total_reward
            metrics = {
                'total_steps': self.total_steps,
                'episode': episode,
                'total_reward': total_reward,
                'running_total_reward': running_total_reward,
            }
            if 'epsilon' in self.agent.__dict__:
                metrics['epsilon'] = self.agent.epsilon
            self.logger.log_metrics(metrics, episode=episode)

        self.logger.finish()

        if self.show_episode:
            show_episode(self.config['environment_name'], self.agent)
