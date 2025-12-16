# TODO: вынести в файлы конфигурации
HEIGHT_REWARD: float = 2.0
BLOCK_GROUPING_REWARD: float = 1.0
COLLISION_BLOCKS_REWARD: float = 1.5


class FakeRewardCalculator:
    def __init__(self):
        self._max_height: float = 0.0
        self._collisioned_blocks: int = 0
        self._block_grouping: float = 0.0  # <= 1.0

    # TODO: добавить параметры для настройки функции наград
    # максимальное увеличение высоты
    # количество упавших блоков
    # стабильность башни
    # кучность блоков
    # максимальная скорость блоков
    def fill_physics(
        self,
        max_height: float,
        collisioned_blocks: int,
        block_grouping: float
    ):
        self._max_height = max_height
        self._collisioned_blocks = collisioned_blocks
        self._block_grouping = block_grouping

    def calculate_reward(self) -> float:
        reward: float = 0.0
        reward += self._max_height * HEIGHT_REWARD
        reward += self._collisioned_blocks * COLLISION_BLOCKS_REWARD
        reward += self._block_grouping * BLOCK_GROUPING_REWARD
        return reward
