import os
import random
import time
from gymnasium import spaces
import numpy as np
import gymnasium as gym
from typing import Any, Dict, List, SupportsFloat, Tuple

import mujoco_warp as mujoco

from reward_calculator import FakeRewardCalculator

XML_FOLDER = "configurations"


def euler_to_quat(roll, pitch, yaw):
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)


class JengaEnv6DoF(gym.Env):
    def __init__(self, n_blocks=10):
        self.n_blocks = n_blocks

        # Полуразмеры блока
        self.half_z = 0.05
        self.half_x = 6 * self.half_z
        self.half_y = 2 * self.half_z

        # Генерация XML
        self.xml_path = self._generate_xml()

        # Загрузка модели и данных
        self.model = mujoco.Model(self.xml_path)
        self.data = mujoco.Data(self.model)

        # Получаем ID блоков
        self.block_ids = [
            self.model.body(f"block_{i}").id
            for i in range(self.n_blocks)
        ]

        self.reward_calculator = FakeRewardCalculator()

        # Храним предыдущее состояние для расчета изменений
        self.prev_heights = np.zeros(n_blocks)
        self.prev_state = None
        self.initial_heights = np.zeros(n_blocks)

    def _get_state_volume(self) -> int:
        return 3 + 4 + 3 + 3  # how many variables in single "state" structure

    def get_state_dim(self) -> int:
        return self.n_blocks * self._get_state_volume()

    def get_action_dims(self) -> List[int]:
        return [
            self.n_blocks,
            11,
            11,
            11,
            11,
            11,
            11,
        ]  # 10 блоков, 11 значений каждой силы и вращения по каждой оси

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
            # случайная позиция блока
            x = float(np.round(np.random.uniform(-0.4, 0.4), 4))
            y = float(np.round(np.random.uniform(-0.4, 0.4), 4))
            z = float(np.round(np.random.uniform(0.02, 0.15), 4))  # чуть выше пола

            # случайный угол в радианах
            roll = float(np.round(np.random.uniform(-np.pi / 4, np.pi / 4), 4))
            pitch = float(np.round(np.random.uniform(-np.pi / 4, np.pi / 4), 4))
            yaw = float(np.round(np.random.uniform(-np.pi, np.pi), 4))

            quat = euler_to_quat(roll, pitch, yaw)
            quat_str = " ".join(map(str, quat.tolist()))

            color = "0.9 0.6 0.3 1" if i < self.n_blocks - 1 else "0.2 0.8 0.2 1"

            body_xml = f'''
    <body name="block_{i}" pos="{x} {y} {z}" quat="{quat_str}">
      <joint name="block_{i}_free" type="free"/>
      <geom name="geom_block_{i}" type="box"
            size="{self.half_x} {self.half_y} {self.half_z}"
            density="600" rgba="{color}"/>
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

        print(f">>> XML создан: {xml_path}")
        return xml_path

    # функция выполняет шаг симуляции, используя входные воздействия action
    # возвращает:
    # observation: np.ndarray - состояние окружения после выполнения шага
    # reward: SupportsFloat - награда за действие
    # terminated: bool - симуляция завершена
    # truncated: bool - время симуляции закончилось
    # info: Dict[str, Any] - дополнительная отладочная информация
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:

        # Сбрасываем приложенные силы
        self.data.qfrc_applied[:] = 0

        # Увеличиваем количество шагов физики
        NUM_PHYSICS_STEPS = 5

        for step_idx in range(NUM_PHYSICS_STEPS):
            # Применяем действие с усилением
            enhanced_action = self._enhance_action(action, step_idx, NUM_PHYSICS_STEPS)
            self._apply_action(enhanced_action)
            mujoco.step(self.model, self.data)

        state = self._get_current_state()

        # Если это первый шаг, сохраняем начальное состояние
        if self.prev_state is None:
            self.prev_state = state.copy()
            self._update_heights_from_state(state, store_initial=True)
        else:
            self._update_heights_from_state(state, store_initial=False)
            self.prev_state = state.copy()

        reward = self._calculate_reward(state)

        terminated = False
        truncated = False
        return state, reward, terminated, truncated, {}

    def _enhance_action(self, action, step_idx, total_steps):
        enhanced_action = action.copy()
        enhanced_action["force"] /= total_steps
        enhanced_action["angular"] /= total_steps
        return enhanced_action

    def _apply_action(self, action):
        # Сбрасываем приложенные силы
        self.data.qfrc_applied[:] = 0

        index = action["block"]
        fx, fy, fz = action["force"]
        tx, ty, tz = action["angular"]

        body = self.model.body(index)
        adr = body.qveladr

        # Применяем силу (первые 3 DOF) и момент (следующие 3 DOF)
        self.data.qfrc_applied[adr:adr + 3] = [fx, fy, fz]
        self.data.qfrc_applied[adr + 3:adr + 6] = [tx, ty, tz]

    # cброс данных окружения
    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        mujoco.reset(self.model, self.data)
        mujoco.forward(self.model, self.data)

        state = self._get_current_state()

        self.prev_state = None
        self.prev_heights = np.zeros(self.n_blocks)
        self.initial_heights = np.zeros(self.n_blocks)

        self._update_heights_from_state(state, store_initial=True)

        return state, {}

    # вспомогательная функция для получения текущего состояния
    def _get_current_state(self) -> np.ndarray:
        state = []
        for body_id in self.block_ids:
            body = self.model.body(body_id)
            qpos_adr = body.qposadr
            qvel_adr = body.qveladr

            state.extend(self.data.qpos[qpos_adr:qpos_adr + 7])
            state.extend(self.data.qvel[qvel_adr:qvel_adr + 6])

        return np.array(state, dtype=np.float32)

    def close(self):
        pass

    def _update_heights_from_state(self, state: np.ndarray, store_initial: bool = False):
        """Обновляет высоты блоков из текущего состояния"""
        for i in range(self.n_blocks):
            z_idx = i * self._get_state_volume() + 2
            current_height = state[z_idx]

            if store_initial:
                self.initial_heights[i] = current_height

            self.prev_heights[i] = current_height

    def _calculate_max_height_change(self, state: np.ndarray) -> float:
        if self.prev_state is None:
            return 0.0

        max_change = 0.0
        for i in range(self.n_blocks):
            z_idx = i * self._get_state_volume() + 2
            height_change = abs(state[z_idx] - self.initial_heights[i])
            max_change = max(max_change, height_change)

        return max_change

    def _calculate_max_block_speed(self, state: np.ndarray) -> float:
        max_speed = 0.0
        for i in range(self.n_blocks):
            vel_start = i * self._get_state_volume() + 7
            speed = np.linalg.norm(state[vel_start:vel_start + 3])
            max_speed = max(max_speed, speed)
        return max_speed

    def _calculate_reward(self, state: np.ndarray) -> float:
        self.reward_calculator.fill_physics(
            max_height_change=self._calculate_max_height_change(state),
            fallen_blocks=0,
            block_grouping=0,
            max_block_speed=self._calculate_max_block_speed(state),
        )
        return self.reward_calculator.calculate_reward()


class ActionWrapper(gym.ActionWrapper):
    """
    Обертка для преобразования Dict action space в MultiDiscrete
    для совместимости со Stable-Baselines3
    """

    def __init__(self, env: JengaEnv6DoF):
        super().__init__(env)

        MOVEMENT_CONSTRAINT = 1
        ROTATION_CONSTRAINT = 0.2

        self.n_force_bins = 21
        self.force_bins = np.linspace(-MOVEMENT_CONSTRAINT, MOVEMENT_CONSTRAINT, self.n_force_bins)
        self.ang_bins = np.linspace(-ROTATION_CONSTRAINT, ROTATION_CONSTRAINT, self.n_force_bins)

        self.action_space = spaces.MultiDiscrete(
            [
                env.n_blocks,
                self.n_force_bins,
                self.n_force_bins,
                self.n_force_bins,
                self.n_force_bins,
                self.n_force_bins,
                self.n_force_bins,
            ]
        )

    def action(self, action):
        block_idx = action[0]

        force_x = self.force_bins[action[1]]
        force_y = self.force_bins[action[2]]
        force_z = self.force_bins[action[3]]

        ang_x = self.ang_bins[action[4]]
        ang_y = self.ang_bins[action[5]]
        ang_z = self.ang_bins[action[6]]

        return {
            "block": block_idx,
            "force": np.array([force_x, force_y, force_z], dtype=np.float32),
            "angular": np.array([ang_x, ang_y, ang_z], dtype=np.float32),
        }


def make_jenga_env(n_blocks: int) -> gym.ActionWrapper:
    random.seed(123)
    np.random.seed(123)

    env = JengaEnv6DoF(n_blocks=n_blocks)
    env = ActionWrapper(env)
    return env
