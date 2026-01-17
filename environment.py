import os
import random
import time
from gymnasium import spaces
import numpy as np

import mujoco

import gymnasium as gym
from mujoco import viewer
from typing import Any, Dict, List, SupportsFloat, Tuple
import scipy
from scipy.spatial.transform import Rotation

from reward_calculator import RewardCalculator

XML_FOLDER = "configurations"

# Константы размеров блока Дженги
BLOCK_LENGTH_X = 0.075  # Длина блока (0.0375 * 2)
BLOCK_LENGTH_Y = 0.025  # Ширина блока (0.0125 * 2)
BLOCK_LENGTH_Z = 0.015  # Высота блока (0.0075 * 2)

# Максимальное перемещение за один шаг
MAX_MOVEMENT_DISTANCE = BLOCK_LENGTH_Z / 3
NUM_STABILIZATION_STEPS = 1

BINS_COUNT = 3

def euler_to_quat(roll, pitch, yaw):
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
    quat = rot.as_quat()  # возвращает [x, y, z, w]

    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)

def quat_to_euler(w, x, y, z):
    # Используем scipy для надежного преобразования
    rot = Rotation.from_quat([x, y, z, w])
    euler = rot.as_euler('xyz', degrees=False)
    return euler[0], euler[1], euler[2]

def get_state_volume__() -> int:
        return 3 + 4  # how many variables in single "state" structure

def jenga_get_state_dim(n_blocks: int) -> int:
    return n_blocks * get_state_volume__()

def jenga_get_action_dims(n_blocks: int) -> List[int]:
    return [
        n_blocks,              # индекс блока
        BINS_COUNT,                 # перемещение по X (-1..1)
        BINS_COUNT,                 # перемещение по Y (-1..1)
        BINS_COUNT,                 # перемещение по Z (-1..1)
        BINS_COUNT,                 # поворот вокруг Y (-pi/4..pi/4)
    ]

class JengaEnv6DoF(gym.Env):
    def __init__(self, n_blocks):
        self.n_blocks = n_blocks

        self.half_x = BLOCK_LENGTH_X / 2
        self.half_y = BLOCK_LENGTH_Y / 2
        self.half_z = BLOCK_LENGTH_Z / 2

        self.xml_path = self._generate_xml()

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        assert(self.model.nbody == self.n_blocks + 1)

        self.block_names = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name.startswith("block_"):
                self.block_names.append(name)

        self.block_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name.encode())
                          for name in self.block_names]

        self.viewer = None
        self._render_counter = 0

        self.reward_calculator = RewardCalculator(block_length=BLOCK_LENGTH_X,
                                                  block_width=BLOCK_LENGTH_Y,
                                                  block_height=BLOCK_LENGTH_Z)

        self.step_count = 0

        obs_dim = jenga_get_state_dim(self.n_blocks)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _generate_xml(self):
        rand_generated = random.randint(0, 1000000)

        xml_name = f"jenga_{rand_generated}.xml"
        if not os.path.exists(XML_FOLDER):
            os.mkdir(XML_FOLDER)

        xml_path = os.path.join(XML_FOLDER, xml_name)
        if os.path.exists(xml_path):
            return xml_path

        header = f"""<mujoco model="jenga_generated">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.85 0.85 0.85 1"/>
"""

        bodies = []

        for i in range(self.n_blocks):
            x = float(np.round(np.random.uniform(-0.4, 0.4), 4))
            y = float(np.round(np.random.uniform(-0.4, 0.4), 4))
            z = float(np.round(np.random.uniform(self.half_x, self.half_x * 2), 4))  # чуть выше пола

            yaw = float(np.round(np.random.uniform(-np.pi, np.pi), 4))

            quat = euler_to_quat(0, 0, yaw)
            quat_str = " ".join(map(str, quat.tolist()))

            color = "0.9 0.6 0.3 1" if i < self.n_blocks - 1 else "0.2 0.8 0.2 1"

            body_xml = f'''
    <body name="block_{i}" pos="{x} {y} {z}" quat="{quat_str}">
      <joint name="block_{i}_free" type="free"/>
      <geom name="geom_block_{i}" type="box" size="{self.half_x} {self.half_y} {self.half_z}" density="600" rgba="{color}"/>
      <site name="site_block_{i}" pos="0 0 0" size="0.002"/>
    </body>
'''
            bodies.append(body_xml)

        footer = """
  </worldbody>
</mujoco>
"""
        xml_text = header + "".join(bodies) + footer
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_text)

        return xml_path

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:

        self.step_count += 1
        self._apply_teleportation(action)

        for _ in range(NUM_STABILIZATION_STEPS):
            mujoco.mj_step(self.model, self.data)

        state = self._get_current_state()
        reward = self._calculate_reward(state)

        done = self._check_done()
        truncated = False

        info = {
            "step": self.step_count,
        }

        return state, reward, done, truncated, info

    def _apply_teleportation(self, action):
        block_idx = action["block"]
        body_id = self.block_ids[block_idx]

        current_pos = self.data.xpos[body_id].copy()
        current_quat = self.data.xquat[body_id].copy()

        desired_displacement = action["force"]

        # Ограничиваем перемещение максимальной дистанцией
        displacement_norm = np.linalg.norm(desired_displacement)
        if displacement_norm > MAX_MOVEMENT_DISTANCE:
            desired_displacement = desired_displacement / displacement_norm * MAX_MOVEMENT_DISTANCE

        new_pos = current_pos + desired_displacement

        current_roll, current_pitch, current_yaw = quat_to_euler(
            current_quat[0], current_quat[1], current_quat[2], current_quat[3]
        )

        desired_angles = action["angular"]

        new_yaw = current_yaw + desired_angles[0]

        new_quat = euler_to_quat(0, 0, new_yaw)

        # Телепортируем блок
        jnt_qpos_index = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
        self.data.qpos[jnt_qpos_index:jnt_qpos_index+3] = new_pos
        self.data.qpos[jnt_qpos_index+3:jnt_qpos_index+7] = new_quat

        jnt_dof_adr = self.model.jnt_dofadr[self.model.body_jntadr[body_id]]
        self.data.qvel[jnt_dof_adr:jnt_dof_adr+6] = 0

        mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)  # важно для gymnasium
        self.reward_calculator.reset()

        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        for body_id in self.block_ids:
            # Находим индекс сустава для этого блока
            jnt_id = self.model.body_jntadr[body_id]
            if jnt_id == -1:
                continue  # пропускаем, если нет сустава

            qpos_idx = self.model.jnt_qposadr[jnt_id]

            pos = self.data.qpos[qpos_idx:qpos_idx+3].copy()
            quat = self.data.qpos[qpos_idx+3:qpos_idx+7].copy()

            pos[0] += np.random.uniform(-0.015, 0.015)  # X
            pos[1] += np.random.uniform(-0.015, 0.015)  # Y
            pos[2] += np.random.uniform(-0.005, 0.015)  # Z (меньше шума по высоте)

            roll, pitch, yaw = quat_to_euler(quat[0], quat[1], quat[2], quat[3])
            yaw += np.random.uniform(-0.087, 0.087)  # ±5 градусов в радианах
            new_quat = euler_to_quat(0, 0, yaw)  # сохраняем только yaw

            # 3. Обновляем состояние
            self.data.qpos[qpos_idx:qpos_idx+3] = pos
            self.data.qpos[qpos_idx+3:qpos_idx+7] = new_quat

            # 4. Обнуляем скорость для стабильности
            dof_idx = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_idx:dof_idx+6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # Стабилизация после изменения позиций
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)

        state = self._get_current_state()
        return state, {}  # gymnasium требует dict

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

        return None

    def debug_output(self):
        block_positions = []

        for body_id in self.block_ids:
            pos = self.data.xpos[body_id].copy()
            block_positions.append(pos)

        print("")
        for i, pos in enumerate(block_positions):
            print(f"block {i}: {pos}")

    def _render_frame(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)

        self.viewer.sync()
        self._render_counter += 1

    def _get_current_state(self) -> np.ndarray:
        state = []
        for body_id in self.block_ids:
            pos = self.data.xpos[body_id]
            quat = self.data.xquat[body_id]
            state.extend(pos)
            state.extend(quat)
        return np.array(state, dtype=np.float32)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _check_done(self) -> bool:
        return len(self.reward_calculator.get_placed_blocks()) == self.n_blocks

    def _calculate_reward(self, state: np.ndarray) -> float:
        block_positions = []

        for body_id in self.block_ids:
            pos = self.data.xpos[body_id].copy()
            block_positions.append(pos)

        self.reward_calculator.fill_physics(block_positions)
        reward = self.reward_calculator.calculate_reward()
        return reward

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env: JengaEnv6DoF, n_blocks: int):
        super().__init__(env)

        self.n_bins = BINS_COUNT
        self.max_movement = MAX_MOVEMENT_DISTANCE
        self.max_angle = np.pi / 4  # 90 градусов

        self.action_space = spaces.MultiDiscrete(jenga_get_action_dims(n_blocks))

        self.disp_bins = np.linspace(-1.0, 1.0, self.n_bins)  # для x, y
        self.angle_bins = np.linspace(-self.max_angle, self.max_angle, self.n_bins)

    def action(self, action: np.ndarray):
        block_idx = int(action[0])

        dx = self.disp_bins[action[1]] * self.max_movement
        dy = self.disp_bins[action[2]] * self.max_movement
        dz = self.disp_bins[action[3]] * self.max_movement

        yaw = self.angle_bins[action[4]]

        return {
            "block": block_idx,
            "force": np.array([dx, dy, dz], dtype=np.float32),
            "angular": np.array([yaw], dtype=np.float32),
        }

    def reverse_action(self, action):
        raise NotImplementedError


def make_jenga_env(n_blocks: int, render: bool = False, seed: int = 123) -> gym.Env:
    env = JengaEnv6DoF(n_blocks=n_blocks)
    env.reset(seed=seed)

    if render:
        env.render_mode = "human"
    env = ActionWrapper(env, n_blocks)
    return env
