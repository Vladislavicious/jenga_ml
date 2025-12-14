import os
import random
import time
import numpy as np
import mujoco
import gymnasium as gym
from mujoco import viewer
from typing import Any, Dict, List, SupportsFloat, Tuple

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
        self.half_x = 0.0375
        self.half_y = 0.0125
        self.half_z = 0.0075

        # Генерация XML
        self.xml_path = self._generate_xml()

        # Загрузка модели и данных
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

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

    def get_state_dim(self) -> int:
        return self.n_blocks

    def get_action_dims(self) -> List[int]:
        return [
            self.n_blocks,
            11,
            11,
            11,
        ]  # 10 блоков, 11 значений каждой силы по каждой оси

    def _generate_xml(self):
        rand_generated = random.randint(0, 1000000)
        xml_name = f"jenga_{rand_generated}.xml"
        if not os.path.exists(XML_FOLDER):
          os.mkdir(XML_FOLDER)

        xml_path = os.path.join(XML_FOLDER, xml_name)

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
        self._apply_action(action)

        # Выполняем несколько шагов физики
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        state = self._get_current_state()
        reward = 0.0
        terminated = False
        truncated = False
        return state, reward, terminated, truncated, {}

    def _apply_action(self, action):
        action = np.array(action).reshape(len(self.block_ids), 6)

        # Сбрасываем приложенные силы
        self.data.qfrc_applied[:] = 0

        for i, body_id in enumerate(self.block_ids):
            # Действие как сила/момент
            fx, fy, fz, tx, ty, tz = action[i]

            # Получаем индекс DOF для этого тела
            jnt_dof_adr = self.model.jnt_dofadr[self.model.body_jntadr[body_id]]

            # Применяем силу (первые 3 DOF) и момент (следующие 3 DOF)
            self.data.qfrc_applied[jnt_dof_adr:jnt_dof_adr+3] = [fx, fy, fz]
            self.data.qfrc_applied[jnt_dof_adr+3:jnt_dof_adr+6] = [tx, ty, tz]

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
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_current_state()

    # функция отображения, делает отображение по свойству render_mode (смотри gym.Env)
    # для простоты отрисовываем только один режим
    def render(self):
        """Визуализация среды"""
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
