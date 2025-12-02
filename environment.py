# there will be all MuJoCo-related stuff


from typing import Any, Dict, List, SupportsFloat, Tuple
import gymnasium as gym

import numpy as np

from reward_calculator import FakeRewardCalculator


class FakeEnvironment(gym.Env):
    def __init__(self):
        self._state_dim = 10

        self.reward_calculator = FakeRewardCalculator()

    def get_state_dim(self) -> int:
        return self._state_dim

    def get_action_dims(self) -> List[int]:
        return [
            self._state_dim,
            11,
            11,
            11,
        ]  # 10 блоков, 11 значений каждой силы по каждой оси

    # функция выполняет шаг симуляции, используя входные воздействия action
    # возвращает:
    # observation: np.ndarray - состояние окружения после выполнения шага
    # reward: SupportsFloat - награда за действие, вычисляется с self.reward_calculator, используя функцию fill_physics
    # terminated: bool - симуляция завершена
    # truncated: bool - ввремя симуляции закончилось
    # info: Dict[str, Any] - дополнительная отладочная информация
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        pass

    # cброс данных окружения (MuJoCo) - координаты каждого блока
    # еще не уверен по поводу размеров блоков и нужны ли они алгоритму обучения
    # seed: int - начальное значение PRNG окружения
    # возвращает нулевое состояние - после совершения сброса
    # info: Dict[str, Any] - дополнительная отладочная информация
    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        pass

    # функция отображения, делает отображение по свойству render_mode (смотри gym.Env)
    # для простоты отрисовываем только один режим
    def render(self, mode: str):
        """Визуализация среды"""
        if self.render_mode == "human":
            self._render_frame()

        return None

    # вспомогательная функция для отображения текущего состояния
    def _render_frame(self):
        pass

    # вспомогательная функция для получения текущего состоянияы
    def _get_current_state(self) -> np.ndarray:
        return np.array()
