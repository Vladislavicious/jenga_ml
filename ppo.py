from typing import Tuple, List

import torch
import torch.nn as nn


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

## constants to move them in future
LEARNING_RATE: float = 3e-4
GAE_LAMBDA: float = 0.95,
GAMMA: float = 0.99,
CLIP_EPSILON: float = 0.2,
ENTROPY_COEF: float = 0.01,
VALUE_COEF: float = 0.5,
MAX_GRAD_NORM: float = 0.5,
DEVICE: str = 'cuda'

# Реализация алгоритма PPO
class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dims: List[int],
    ):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        self.gamma = GAMMA
        self.gae_lambda = GAE_LAMBDA
        self.clip_epsilon = CLIP_EPSILON
        self.entropy_coef = ENTROPY_COEF
        self.value_coef = VALUE_COEF
        self.max_grad_norm = MAX_GRAD_NORM

        self.network: PPONetwork = PPONetwork(state_dim, action_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
