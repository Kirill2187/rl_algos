from src.agents.agent import Agent
import gymnasium as gym
import threading


def show_episode(env_name: str, agent: Agent, max_steps: int = 1000):
    def render():
        env = gym.make(env_name, render_mode='human')
        state, _ = env.reset()
        for step in range(max_steps):
            action = agent.select_action(state, mode='eval')
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()

    thread = threading.Thread(target=render)
    thread.start()
