from typing import List
from environment import FakeEnvironment
from ppo import PPOAgent


class JengaML_Trainer:
    def __init__(
        self,
        model_environment: FakeEnvironment,
        blocks_count: int,
        total_timesteps: int,
        n_steps: int,
    ):
        self.model_environment: FakeEnvironment = model_environment
        self.blocks_count = blocks_count

        # Параметры обучения
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps

        # Создание агента
        state_dimensions: int = self.model_environment.get_state_dim()
        action_dimensions: List[int] = model_environment.get_action_dims()

        self.agent = PPOAgent()
        self.agent.initialize_network(
            state_dim=state_dimensions,
            action_dims=action_dimensions,
        )

    def train(self):
        pass

    def evaluate(self, visualize: bool = False):
        pass
