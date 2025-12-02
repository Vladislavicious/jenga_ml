# there will be all MuJoCo-related stuff


from typing import List, Tuple

import numpy as np


class FakeRewardCalculator:
    def __init__(self):
        pass

    # TODO: добавить параметры для настройки функции наград
    # максимальное увеличение высоты
    # количество упавших блоков
    # стабильность башни
    # кучность блоков
    # максимальная скорость блоков
    def fill_physics(self):
        pass

    def calculate_reward(self) -> float:
        pass


class FakeEnvironment:
    def __init__(self):
        self._state_dim = 10

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
    # по окончании action, заполняет параметрами reward_calculator, используя функцию fill_physics
    # возвращает:
    # next_state: np.ndarray - состояние окружения после выполнения шага
    # simulation_end: bool - симуляция аварийно завершена/башня упала = True
    def step(
        self, action: np.ndarray, reward_calculator: FakeRewardCalculator
    ) -> Tuple[np.ndarray, bool]:
        pass

    # cброс данных окружения (MuJoCo) - координаты каждого блока
    # еще не уверен по поводу размеров блоков и нужны ли они алгоритму обучения
    # возвращает нулевое состояние - после совершения сброса
    def reset(self) -> np.ndarray:
        pass

    # вспомогательная функция для получения текущего состоянияы
    def _get_current_state(self) -> np.ndarray:
        return np.array()
