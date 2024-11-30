from abc import ABC, abstractmethod
from typing import Literal


class Agent(ABC):
    @abstractmethod
    def select_action(self, state, mode: Literal['train', 'eval']):
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def after_episode(self, episode):
        pass


