from typing import Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

# TODO: переместить в другое место
LEARNING_RATE: float = 3e-4
GAE_LAMBDA: float = (0.95,)
GAMMA: float = (0.99,)
CLIP_EPSILON: float = (0.2,)
ENTROPY_COEF: float = (0.01,)
VALUE_COEF: float = (0.5,)
MAX_GRAD_NORM: float = (0.5,)
DEVICE: str = "cuda"


# Нейронная сеть для PPO
class PPONetwork(nn.Module):
    def __init__(
        self,
        state_dim_count: int,
        action_dims_count: List[int],
        hidden_dim_count: int = 256,
    ):
        super().__init__()

        self.state_dim_count = state_dim_count  # Размерность вектора состояния
        self.action_dims_count = action_dims_count  # Размерности векторов для каждого компонента действия (выбор блока, сила по X, сила по Y, сила по Z)

        self.shared_features = nn.Sequential(
            nn.Linear(self.state_dim_count, hidden_dim_count),
            nn.Tanh(),  # Функция активации
            nn.Linear(hidden_dim_count, hidden_dim_count),
            nn.Tanh(),
        )

        self.actor_heads = nn.ModuleList(
            [nn.Linear(hidden_dim_count, dim) for dim in self.action_dims_count]
        )
        self.critic_head = nn.Linear(hidden_dim_count, 1)

    def forward(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        features = self.shared_features(state)

        action_logits = [head(features) for head in self.actor_heads]
        value = self.critic_head(features)

        return action_logits, value


# Реализация алгоритма PPO
class PPOAgent:
    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.gamma = GAMMA
        self.gae_lambda = GAE_LAMBDA
        self.clip_epsilon = CLIP_EPSILON
        self.entropy_coef = ENTROPY_COEF
        self.value_coef = VALUE_COEF
        self.max_grad_norm = MAX_GRAD_NORM

    # Выбор действия для заданного состояния
    def get_action(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, value = self.network(state_tensor)

            # Сэмплируем действия из каждого распределения
            actions = []
            log_probs = []

            for logits in action_logits:
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                actions.append(action.item())
                log_probs.append(log_prob)

            total_log_prob = torch.stack(log_probs).sum()

            return (
                np.array(actions),
                total_log_prob.cpu().numpy(),
                value.squeeze().cpu().numpy(),
            )

    def compute_advantages(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = []
        returns = []

        advantage = 0
        next_value = next_value

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            # Оценка обобщенного преимущества
            advantage = delta + self.gamma * self.gae_lambda * advantage * (
                1 - dones[t]
            )
            advantages.insert(0, advantage)

            returns.insert(0, advantage + values[t])
            next_value = values[t]

        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)

        # Нормализация
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        epochs: int,
        batch_size: int,
    ):
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        dataset = TensorDataset(
            states_tensor,
            actions_tensor,
            old_log_probs_tensor,
            advantages_tensor,
            returns_tensor,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(epochs):
            for batch in dataloader:
                (
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                ) = batch

                action_logits, values = self.network(batch_states)
                values = values.squeeze()

                new_log_probs = []
                entropies = []

                batch_actions = batch_actions.T  # [batch_size, 4] -> [4, batch_size]

                for i, logits in enumerate(action_logits):
                    dist = Categorical(logits=logits)
                    action = batch_actions[i]
                    new_log_prob = dist.log_prob(action)
                    entropy = dist.entropy()

                    new_log_probs.append(new_log_prob)
                    entropies.append(entropy)

                new_log_probs = torch.stack(new_log_probs, dim=1).sum(
                    dim=1
                )  # Суммируем логарифмы вероятностей по компонентам действий
                entropy = torch.stack(entropies, dim=1).sum(dim=1).mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)

                # Общий loss и оптимизация
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        # Средние значения по всем обновлениям
        n_updates = epochs * len(dataloader)
        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy = total_entropy / n_updates

        return avg_policy_loss, avg_value_loss, avg_entropy

    def save(self, path: str):
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
