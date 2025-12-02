import torch
import torch.nn as nn
from typing import Tuple, List

# Нейронная сеть для PPO
class PPONetwork(nn.Module):
    def __init__(self, state_dim_count: int, action_dims_count: List[int],
                 hidden_dim_count: int = 256):
        super().__init__()

        self.state_dim_count = state_dim_count # Размерность вектора состояния
        self.action_dims_count = action_dims_count  # Размерности векторов для каждого компонента действия (выбор блока, сила по X, сила по Y, сила по Z)

        self.shared_features = nn.Sequential(
            nn.Linear(self.state_dim_count, hidden_dim_count),
            nn.Tanh(), # Функция активации
            nn.Linear(hidden_dim_count, hidden_dim_count),
            nn.Tanh(),
        )

        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_dim_count, dim) for dim in self.action_dims_count
        ])
        self.critic_head = nn.Linear(hidden_dim_count, 1)

    def forward(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        features = self.shared_features(state)

        action_logits = [head(features) for head in self.actor_heads]
        value = self.critic_head(features)

        return action_logits, value
