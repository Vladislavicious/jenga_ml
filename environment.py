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

from reward_calculator import FakeRewardCalculator

XML_FOLDER = "configurations"

# Константы размеров блока Дженги
BLOCK_LENGTH_X = 0.075  # Длина блока (0.0375 * 2)
BLOCK_LENGTH_Y = 0.025  # Ширина блока (0.0125 * 2)
BLOCK_LENGTH_Z = 0.015  # Высота блока (0.0075 * 2)

# Максимальное перемещение за один шаг
MAX_MOVEMENT_DISTANCE = BLOCK_LENGTH_Y
NUM_STABILIZATION_STEPS = 10

def euler_to_quat(roll, pitch, yaw):
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
    quat = rot.as_quat()  # возвращает [x, y, z, w]

    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)

def quat_to_euler(w, x, y, z):
    # Используем scipy для надежного преобразования
    rot = Rotation.from_quat([x, y, z, w])
    euler = rot.as_euler('xyz', degrees=False)
    return euler[0], euler[1], euler[2]

def are_blocks_touching(pos1: np.ndarray, ori1: np.ndarray, size1: np.ndarray,
                        pos2: np.ndarray, ori2: np.ndarray, size2: np.ndarray,
                        tolerance: float = 1e-3) -> bool:
    half_size1 = size1 / 2.0
    half_size2 = size2 / 2.0

    # Проверка пересечения по каждой оси
    for i in range(3):
        if abs(pos1[i] - pos2[i]) > (half_size1[i] + half_size2[i] + tolerance):
            return False

    return True


class JengaEnv6DoF(gym.Env):
    def __init__(self, n_blocks):
        self.n_blocks = n_blocks

        # Полуразмеры блока (используем для расчетов)
        self.half_x = BLOCK_LENGTH_X / 2
        self.half_y = BLOCK_LENGTH_Y / 2
        self.half_z = BLOCK_LENGTH_Z / 2
        self.block_size = np.array([BLOCK_LENGTH_X, BLOCK_LENGTH_Y, BLOCK_LENGTH_Z])

        # Генерация XML
        self.xml_path = self._generate_xml()

        # Загрузка модели и данных
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        assert(self.model.nbody == self.n_blocks + 1)

        # Получаем имена и ID блоков
        self.block_names = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name.startswith("block_"):
                self.block_names.append(name)

        self.block_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name.encode())
                          for name in self.block_names]

        self.viewer = None
        self._render_counter = 0

        self.reward_calculator = FakeRewardCalculator()

        # Для дебаггинга
        self.step_count = 0

    def _get_state_volume(self) -> int:
        return 3 + 4 + 3 + 3  # how many variables in single "state" structure

    def get_state_dim(self) -> int:
        return self.n_blocks * self._get_state_volume()

    def get_action_dims(self) -> List[int]:
        return [
            self.n_blocks,      # индекс блока
            21,                 # перемещение по X (-1..1)
            21,                 # перемещение по Y (-1..1)
            21,                 # перемещение по Z (-1..1)
            21,                 # поворот вокруг X (-pi/4..pi/4)
            21,                 # поворот вокруг Y (-pi/4..pi/4)
            21,                 # поворот вокруг Z (-pi/2..pi/2)
        ]

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
            roll = float(np.round(np.random.uniform(-np.pi/4, np.pi/4), 4))
            pitch = float(np.round(np.random.uniform(-np.pi/4, np.pi/4), 4))
            yaw = float(np.round(np.random.uniform(-np.pi, np.pi), 4))

            quat = euler_to_quat(roll, pitch, yaw)
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
        print(f">>> XML создан: {xml_path}")
        return xml_path

    def _compute_max_height(self) -> float:
        max_height = -10000.0
        for body_id in self.block_ids:
            height = self.data.xpos[body_id][2]
            if height > max_height:
                max_height = height
        return max_height

    def _count_collisioned_blocks(self) -> int:
        collisioned = set()

        # Получаем позиции и ориентации всех блоков
        block_positions = []
        block_orientations = []

        for body_id in self.block_ids:
            pos = self.data.xpos[body_id].copy()
            quat = self.data.xquat[body_id].copy()
            block_positions.append(pos)
            block_orientations.append(quat)

        # Проверяем все пары блоков
        for i in range(self.n_blocks):
            for j in range(i + 1, self.n_blocks):
                if are_blocks_touching(
                    block_positions[i], block_orientations[i], self.block_size,
                    block_positions[j], block_orientations[j], self.block_size
                ):
                    collisioned.add(i)
                    collisioned.add(j)

        return len(collisioned)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:

        self.step_count += 1

        # Применяем телепортацию
        self._apply_teleportation(action)

        # Выполняем несколько шагов физики для стабилизации
        for _ in range(NUM_STABILIZATION_STEPS):
            mujoco.mj_step(self.model, self.data)

        # Получаем состояние после стабилизации
        state = self._get_current_state()

        # Вычисляем награду
        reward = self._calculate_reward(state)

        terminated = self._check_termination(state)
        truncated = False

        info = {
            "step": self.step_count,
        }

        return state, reward, terminated, truncated, info

    def _apply_teleportation(self, action):
        block_idx = action["block"]
        body_id = self.block_ids[block_idx]

        # Получаем текущую позицию и ориентацию
        current_pos = self.data.xpos[body_id].copy()
        current_quat = self.data.xquat[body_id].copy()

        # Преобразуем желаемое перемещение
        desired_displacement = action["force"]

        # Ограничиваем перемещение максимальной дистанцией
        displacement_norm = np.linalg.norm(desired_displacement)
        if displacement_norm > MAX_MOVEMENT_DISTANCE:
            desired_displacement = desired_displacement / displacement_norm * MAX_MOVEMENT_DISTANCE

        # Вычисляем новую позицию
        new_pos = current_pos + desired_displacement

        # Преобразуем углы поворота
        current_roll, current_pitch, current_yaw = quat_to_euler(
            current_quat[0], current_quat[1], current_quat[2], current_quat[3]
        )

        # Ограничиваем углы поворота
        max_angle = np.pi / 4  # 45 градусов
        desired_angles = action["angular"]
        desired_angles = np.clip(desired_angles, -max_angle, max_angle)

        # Вычисляем новую ориентацию
        new_roll = current_roll + desired_angles[0]
        new_pitch = current_pitch + desired_angles[1]
        new_yaw = current_yaw + desired_angles[2]

        new_quat = euler_to_quat(new_roll, new_pitch, new_yaw)

        # Телепортируем блок
        jnt_qpos_adr = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]

        # Устанавливаем новую позицию
        self.data.qpos[jnt_qpos_adr:jnt_qpos_adr+3] = new_pos

        # Устанавливаем новую ориентацию
        self.data.qpos[jnt_qpos_adr+3:jnt_qpos_adr+7] = new_quat

        # Обнуляем скорости
        jnt_dof_adr = self.model.jnt_dofadr[self.model.body_jntadr[body_id]]
        self.data.qvel[jnt_dof_adr:jnt_dof_adr+6] = 0

        # Обновляем физику
        mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        state = self._get_current_state()

        return state

    # функция отображения, делает отображение по свойству render_mode (смотри gym.Env)
    # для простоты отрисовываем только один режим
    def render(self):
        if self.render_mode == "human":
            self._render_frame()

        return None

    # вспомогательная функция для отображения текущего состояния
    def _render_frame(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)

        self.viewer.sync()
        self._render_counter += 1

    # вспомогательная функция для получения текущего состоянияы
    def _get_current_state(self) -> np.ndarray:
        state = []
        for body_id in self.block_ids:
            pos = self.data.xpos[body_id]
            quat = self.data.xquat[body_id]
            lin_vel = self.data.cvel[body_id][:3]
            ang_vel = self.data.cvel[body_id][3:]
            state.extend(pos)
            state.extend(quat)
            state.extend(lin_vel)
            state.extend(ang_vel)
        return np.array(state, dtype=np.float32)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _calculate_grouping_coef(self, state: np.ndarray) -> float:
        distances = []

        for i in range(self.n_blocks):
            x_idx = i * self._get_state_volume()
            y_idx = x_idx + 1

            x_pos = state[x_idx]
            y_pos = state[y_idx]

            distance = np.sqrt(x_pos**2 + y_pos**2)
            distances.append(distance)

        mean_distance = np.mean(distances)

        return (BLOCK_LENGTH_X * 3) - mean_distance

    def _check_termination(self, state: np.ndarray) -> bool:
        # Проверяем, не упали ли блоки слишком низко
        return False

    def _calculate_reward(self, state: np.ndarray) -> float:
        block_grouping = self._calculate_grouping_coef(state)
        max_height = self._compute_max_height()
        collisioned_blocks = self._count_collisioned_blocks()

        self.reward_calculator.fill_physics(
            max_height=max_height,
            block_grouping=block_grouping,
            collisioned_blocks=collisioned_blocks)

        reward = self.reward_calculator.calculate_reward()
        return reward


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env: JengaEnv6DoF):
        super().__init__(env)

        # Диапазоны для перемещения (в долях от MAX_MOVEMENT_DISTANCE)
        DISPLACEMENT_RANGE = 1.0  # от -1 до 1 относительно MAX_MOVEMENT_DISTANCE

        # Диапазоны для вращения (в радианах)
        ROLL_PITCH_RANGE = np.pi / 4  # ±45 градусов
        YAW_RANGE = np.pi / 2  # ±90 градусов

        self.n_bins = 21

        # Биннинг для перемещения
        self.displacement_bins = np.linspace(
            -DISPLACEMENT_RANGE,
            DISPLACEMENT_RANGE,
            self.n_bins
        )

        # Биннинг для вращения
        self.roll_pitch_bins = np.linspace(
            -ROLL_PITCH_RANGE,
            ROLL_PITCH_RANGE,
            self.n_bins
        )

        self.yaw_bins = np.linspace(
            -YAW_RANGE,
            YAW_RANGE,
            self.n_bins
        )

        self.simulation = env

    def action(self, action):
        block_idx = action[0]

        # Преобразуем дискретные значения в непрерывные перемещения
        # Умножаем на MAX_MOVEMENT_DISTANCE для получения фактического перемещения
        displacement_x = self.displacement_bins[action[1]] * MAX_MOVEMENT_DISTANCE
        displacement_y = self.displacement_bins[action[2]] * MAX_MOVEMENT_DISTANCE
        displacement_z = self.displacement_bins[action[3]] * MAX_MOVEMENT_DISTANCE

        # Углы поворота
        roll = self.roll_pitch_bins[action[4]]
        pitch = self.roll_pitch_bins[action[5]]
        yaw = self.yaw_bins[action[6]]

        return {
            "block": block_idx,
            "force": np.array([displacement_x, displacement_y, displacement_z], dtype=np.float32),
            "angular": np.array([roll, pitch, yaw], dtype=np.float32),
        }

    def get_state_dim(self) -> int:
        return self.simulation.get_state_dim()

    def get_action_dims(self) -> List[int]:
        return self.simulation.get_action_dims()


def make_jenga_env(n_blocks: int, render: bool = False) -> gym.ActionWrapper:
    env = JengaEnv6DoF(n_blocks=n_blocks)
    if render:
        env.render_mode = "human"
    env = ActionWrapper(env)
    return env
