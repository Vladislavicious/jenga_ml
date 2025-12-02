# TODO: вынести в файлы конфигурации
HEIGHT_CHANGE_REWARD: float = 10.0
BLOCK_GROUPING_REWARD: float = 0.5

FALLEN_BLOCKS_PENALTY: float = -2.0
MAX_BLOCK_SPEED_PENALTY: float = 0.5


class FakeRewardCalculator:
    def __init__(self):
        self._max_height_change: float = 0.0
        self._fallen_blocks: int = 0
        self._block_grouping: float = 0.0  # <= 1.0
        self._max_block_speed: float = 0.0

    # TODO: добавить параметры для настройки функции наград
    # максимальное увеличение высоты
    # количество упавших блоков
    # стабильность башни
    # кучность блоков
    # максимальная скорость блоков
    def fill_physics(
        self,
        max_height_change: float,
        fallen_blocks: int,
        block_grouping: float,
        max_block_speed: float,
    ):
        self._max_height_change = max_height_change
        self._fallen_blocks = fallen_blocks
        self._block_grouping = block_grouping
        self._max_block_speed = max_block_speed

    def calculate_reward(self) -> float:
        reward: float = 0.0
        reward += self._max_height_change * HEIGHT_CHANGE_REWARD
        reward += self._fallen_blocks * FALLEN_BLOCKS_PENALTY
        reward += self._block_grouping * BLOCK_GROUPING_REWARD
        reward += self._max_block_speed * MAX_BLOCK_SPEED_PENALTY
        return reward
