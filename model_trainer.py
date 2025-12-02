from typing import Dict, List

import numpy as np
from torch import Tensor
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

        state_dimensions: int = self.model_environment.get_state_dim()
        action_dimensions: List[int] = model_environment.get_action_dims()

        self.agent = PPOAgent()
        self.agent.initialize_network(
            state_dim=state_dimensions,
            action_dims=action_dimensions,
        )

    def train(self):
        """Основной цикл обучения"""
        print(f"Начало обучения на {self.total_timesteps} шагов")

        timestep = 0

        while timestep < self.total_timesteps:
            # Сбор траектории
            step_data = self.perform_train_step()
            timestep += len(step_data["states"])

            # Вычисление преимуществ
            advantages, returns = self.agent.compute_advantages(
                rewards=step_data["rewards"],
                values=step_data["values"],
                dones=step_data["dones"],
                next_value=step_data["last_value"],
            )

            # Обновление политики
            policy_loss, value_loss, entropy = self.agent.update(
                states=step_data["states"],
                actions=step_data["actions"],
                old_log_probs=step_data["log_probs"],
                advantages=advantages,
                returns=returns,
                epochs=10,
                batch_size=64,
            )

        print(f"Всего эпизодов: {len(self.episode_rewards)}")
        print(
            f"Средняя награда за последние 10 эпизодов: {np.mean(self.episode_rewards[-10:]):.2f}"
        )

    def perform_train_step(self) -> Dict:
        """Сбор траектории (опыта)"""
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        last_value = Tensor()

        step_data = {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int32),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=bool),
            "values": np.array(values, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "last_value": last_value,
        }

        return step_data

    def evaluate(self, visualize: bool = False):
        pass
