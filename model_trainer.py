from typing import Dict, List

import numpy as np
from environment import FakeRewardCalculator
import gymnasium as gym
from ppo import PPOAgent

EVALUATION_STEP_COUNT: int = 5


class JengaML_Trainer:
    def __init__(
        self,
        model_environment: gym.ActionWrapper,
        blocks_count: int,
        total_timesteps: int,
        n_steps: int,
        render: bool = True
    ):
        self.model_environment: gym.ActionWrapper = model_environment
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

        self.episode_rewards = []
        self.episode_lengths = []
        self.render = render

    def train(self):
        """Основной цикл обучения"""
        print(f"Начало обучения на {self.total_timesteps} шагов")

        timestep = 0

        while timestep < self.total_timesteps:
            step_data = self.perform_train_step()
            timestep += len(step_data["states"])

            advantages, returns = self.agent.compute_advantages(
                rewards=step_data["rewards"],
                values=step_data["values"],
                dones=step_data["dones"],
                next_value=step_data["last_action_value"],
            )

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

        state = self.model_environment.reset()
        episode_reward = 0
        episode_length = 0

        for _ in range(self.n_steps):
            action, log_prob, value = self.agent.get_action(state)

            next_state, reward, done, truncated, info = self.model_environment.step(
                action
            )

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done or truncated)
            values.append(value)
            log_probs.append(log_prob)

            state = next_state
            episode_reward += reward
            episode_length += 1

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            if done or truncated:

                state = self.model_environment.reset()
                episode_reward = 0
                episode_length = 0

        _, _, last_action_value = self.agent.get_action(state)

        step_data = {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int32),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=bool),
            "values": np.array(values, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "last_action_value": last_action_value,
        }

        return step_data

    def evaluate(self, visualize: bool = False):
        """Оценка обученной модели"""
        print(f"\nОценка модели на {EVALUATION_STEP_COUNT} эпизодах...")

        for episode in range(EVALUATION_STEP_COUNT):
            state = self.model_environment.reset()
            done = False
            truncated = False
            total_reward = 0
            max_height = 0

            step_counter = 0

            for _ in range(100000):
                action, _, _ = self.agent.get_action(state)
                state, reward, done, truncated, info = self.model_environment.step(
                    action
                )
                total_reward += reward
                step_counter = step_counter + 1

                if visualize and step_counter % 60 == 0:
                    self.model_environment.render()

                if "max_height" in info.keys():
                    max_height = max(max_height, info["max_height"])
                    print(f"Макс. высота = {max_height:.2f}")

                if done or truncated:
                    break

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(max_height)

            print(f"Эпизод {episode + 1}: Награда = {total_reward:.2f}")

            print(
                f"\nСредняя награда: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}"
            )
            print(
                f"Средняя макс. высота: {np.mean(self.episode_lengths):.2f} ± {np.std(self.episode_lengths):.2f}"
            )
