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

    def displacement(self) -> np.ndarray:
        return self.current_coords - self.previous_coords

    def movement_magnitude(self) -> float:
        return np.linalg.norm(self.displacement())


class RewardCalculator:
    def __init__(
        self,
        block_length: float,
        block_width: float,
        block_height: float,
        reward_scale: float = 1.0,
        sigma_proximity: float = 0.3,
        attraction_threshold: Optional[float] = None,
        placement_bonus: float = 5.0,
        movement_penalty_coeff: float = 0.1,
        xy_alignment_bonus: float = 2.0,
        max_reward_clip: Optional[float] = None,
    ):
        self.__block_length = block_length
        self.__block_width = block_width
        self.__block_height = block_height

        self.blocks: List[BlockData] = []
        self.current_block_index: int = -1
        self.attraction_point: np.ndarray = np.array([0.0, 0.0, self.__block_height / 2])
        self.placed_blocks: List[int] = []

        self.ATTRACTION_THRESHOLD = attraction_threshold or (0.3 * self.__block_height)

        self.REWARD_SCALE = reward_scale
        self.SIGMA_PROXIMITY = sigma_proximity
        self.PLACEMENT_BONUS = placement_bonus
        self.MOVEMENT_PENALTY_COEFF = movement_penalty_coeff
        self.XY_ALIGNMENT_BONUS = xy_alignment_bonus
        self.MAX_REWARD_CLIP = max_reward_clip

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
        if not self.blocks or self.current_block_index == -1:
            return 0.0

        current_block = self.get_current_block()

        # Проксимальная награда
        distance_to_tp = np.linalg.norm(current_block.current_coords - self.attraction_point)
        proximity_reward = np.exp(-distance_to_tp / self.SIGMA_PROXIMITY)

        if distance_to_tp > 1.0:  # жёсткий порог "слишком далеко"
            proximity_reward = -0.2

        xy_distance = np.linalg.norm(
            current_block.current_coords[:2] - self.attraction_point[:2]
        )
        z_distance = abs(current_block.current_coords[2] - self.attraction_point[2])

        alignment_bonus = 0.0
        if z_distance < 0.1 * self.__block_height and xy_distance < 0.1 * min(self.__block_length, self.__block_width):
            alignment_bonus = self.XY_ALIGNMENT_BONUS

        movement_penalty = 0.0
        if distance_to_tp < self.ATTRACTION_THRESHOLD * 2:
            movement_penalty = -self.MOVEMENT_PENALTY_COEFF * current_block.movement_magnitude()

        placement_bonus = 0.0
        if self._should_select_next_attraction_point():
            self._select_next_attraction_point()
            placement_bonus = self.PLACEMENT_BONUS
            print(f"Block_{self.current_block_index} placed successfully!")

        total_reward = (
            self.REWARD_SCALE * proximity_reward
            + alignment_bonus
            + movement_penalty
            + placement_bonus
        )

        if self.MAX_REWARD_CLIP is not None:
            total_reward = np.clip(total_reward, -self.MAX_REWARD_CLIP, self.MAX_REWARD_CLIP)

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
