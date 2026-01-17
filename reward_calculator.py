import numpy as np

from typing import List, Optional
from dataclasses import dataclass

@dataclass
class BlockData:
    initial_coords: np.ndarray
    current_coords: np.ndarray
    previous_coords: np.ndarray
    is_placed: bool = False

    def update_coords(self, new_coords: np.ndarray) -> None:
        self.previous_coords = np.array(self.current_coords)
        self.current_coords = np.array(new_coords)


class RewardCalculator:
    def __init__(self, block_length: float, block_width: float, block_height: float):
        self.__block_length = block_length
        self.__block_width = block_width
        self.__block_height = block_height

        self.blocks: List[BlockData] = []
        self.current_block_index: int = -1
        self.attraction_point: np.ndarray = np.array([0.0, 0.0, self.__block_height / 2])
        self.placed_blocks: List[int] = []

        # Порог для "захвата" блока (немного больше половины высоты блока)
        self.ATTRACTION_THRESHOLD = 0.3 * self.__block_height

        self.REWARD_SCALE = 1.0  # можно настраивать

        self.is_initialized = False

    def fill_physics(self, block_coords: List[np.ndarray]) -> None:

        if not self.is_initialized:
            for coords in block_coords:
                block = BlockData(
                    initial_coords=np.array(coords),
                    current_coords=np.array(coords),
                    previous_coords=np.array(coords)
                )
                self.blocks.append(block)
            self.is_initialized = True
            self._select_next_current_block()
        else:
            for i, (block, new_coords) in enumerate(zip(self.blocks, block_coords)):
                block.update_coords(new_coords)

    def calculate_reward(self) -> float:
        current_block = self.get_current_block()

        # Абсолютное расстояние до текущей Точки Притяжения
        distance_to_tp = np.linalg.norm(current_block.current_coords - self.attraction_point)

        # Экспоненциальная награда: чем ближе — тем больше
        # Используем exp(-k * d), чтобы избежать взрыва на d=0
        # Но лучше — вознаграждать за малое расстояние: reward = exp(-distance / sigma)
        sigma = 0.4  # характерная ширина "зоны интереса"
        proximity_reward = np.exp(-distance_to_tp / sigma)

        if proximity_reward < 0.003:
            proximity_reward = -0.5

        placement_bonus = 0.0
        if self._should_select_next_attraction_point():
            self._select_next_attraction_point()
            placement_bonus = 5.0 * self.REWARD_SCALE
            print(f"block_{self.current_block_index} placed")

        total_reward = self.REWARD_SCALE * proximity_reward + placement_bonus
        return total_reward

    def _should_select_next_attraction_point(self) -> bool:
        current_block = self.get_current_block()

        if current_block.is_placed:
            return False

        distance = np.linalg.norm(current_block.current_coords - self.attraction_point)
        return distance < self.ATTRACTION_THRESHOLD

    def _select_next_attraction_point(self):
        current_block = self.get_current_block()
        current_block.is_placed = True
        self.placed_blocks.append(self.current_block_index)

        self.attraction_point = current_block.current_coords.copy()
        self.attraction_point[2] += self.__block_height

        self._select_next_current_block()

    def _select_next_current_block(self) -> None:
        min_distance = float('inf')
        closest_index = -1

        for i, block in enumerate(self.blocks):
            if block.is_placed:
                continue

            distance = np.linalg.norm(block.current_coords - self.attraction_point)

            if distance < min_distance:
                min_distance = distance
                closest_index = i

        if closest_index < 0:
            return

        self.current_block_index = closest_index

    def get_current_block(self) -> BlockData:
        return self.blocks[self.current_block_index]

    def get_placed_blocks(self) -> List[int]:
        return self.placed_blocks

    def reset(self) -> None:
        self.blocks.clear()
        self.current_block_index = -1
        self.attraction_point = np.array([0.0, 0.0, self.__block_height / 2])
        self.placed_blocks.clear()
        self.is_initialized = False
