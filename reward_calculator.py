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
        self.current_tower_height: float = 0.0
        self.placed_blocks: List[int] = []

        self.ATTRACTION_THRESHOLD = 0.2 * self.__block_height
        self.DISTANCE_COEFFICIENT = 1000.0  # Коэффициент сближения

        self.is_initialized = False
        self.previous_attraction_point: Optional[np.ndarray] = None

    def fill_physics(self, block_coords: List[np.ndarray]) -> None:

        self.previous_attraction_point = self.attraction_point.copy() if self.is_initialized else None

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

        previous_distance_to_TP = np.linalg.norm(
            current_block.previous_coords - self.attraction_point
        )

        current_distance_to_TP = np.linalg.norm(
            current_block.current_coords - self.attraction_point
        )

        distance_change = previous_distance_to_TP - current_distance_to_TP

        reward = distance_change * self.DISTANCE_COEFFICIENT

        if self._should_select_next_attraction_point():
            self._select_next_attraction_point()

        return reward

    def _should_select_next_attraction_point(self) -> bool:
        current_block = self.get_current_block()

        distance = np.linalg.norm(current_block.current_coords - self.attraction_point)

        if current_block.is_placed:
            return False

        return distance < self.ATTRACTION_THRESHOLD

    def _select_next_attraction_point(self):
        current_block = self.get_current_block()
        current_block.is_placed = True
        self.placed_blocks.append(self.current_block_index)

        self.attraction_point = self._calculate_new_attraction_point()
        self.current_tower_height += self.__block_height

        self._select_next_current_block()

    def _calculate_new_attraction_point(self) -> np.ndarray:
        current_block = self.get_current_block()

        new_attraction_point = current_block.current_coords.copy()
        new_attraction_point[2] += self.__block_height

        return new_attraction_point

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
          print("bad current block selection")

        self.current_block_index = closest_index

    def get_current_block(self) -> BlockData:
        return self.blocks[self.current_block_index]

    def get_placed_blocks(self) -> List[int]:
        return self.placed_blocks

    def reset(self) -> None:
        self.blocks.clear()
        self.current_block_index = -1
        self.attraction_point = np.array([0.0, 0.0, self.__block_height / 2])
        self.current_tower_height = 0.0
        self.placed_blocks.clear()
        self.is_initialized = False
        self.previous_attraction_point = None
