# jenga_env_6dof.py
import os
import random
import time
import numpy as np
import mujoco
from mujoco import viewer

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

class JengaEnv6DoF:
    def __init__(self, n_blocks=10, render_fps=10, seed=None):
        self.n_blocks = n_blocks
        self.render_fps = render_fps
        self.seed = seed if seed is not None else int(time.time()) % 2**32
        np.random.seed(self.seed)

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

    # Получение состояния всех блоков
    def get_state(self):
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

    def apply_action(self, action):
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

    def step(self, action):
        self.apply_action(action)

        # Выполняем несколько шагов физики
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._render_frame()
        state = self.get_state()
        reward = 0.0
        terminated = False
        truncated = False
        return state, reward, terminated, truncated, {}

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def _render_frame(self):
        if self.render_fps == 0:
            return

        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)

        if self._render_counter % self.render_fps == 0:
            self.viewer.sync()
        self._render_counter += 1

        time.sleep(0.03)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    # Создаем окружение
    env = JengaEnv6DoF(n_blocks=10, render_fps=10)  # render_fps=10 для плавной анимации

    # Сбрасываем окружение
    state = env.reset()

    # Выводим начальные позиции первого блока
    print("=== Начальное состояние ===")
    print(f"Позиция блока 0: x={state[0]:.4f}, y={state[1]:.4f}, z={state[2]:.4f}")
    print(f"Ориентация блока 0: w={state[3]:.4f}, x={state[4]:.4f}, y={state[5]:.4f}, z={state[6]:.4f}")

    POS_TEST_STEPS = 100
    ROT_TEST_STEPS = 1000

    if POS_TEST_STEPS != 0:
      print("\n=== Тест изменения позиции ===")
      action_forward = np.zeros((10, 6), dtype=np.float32)
      action_forward[:, 0] = 0.05  # fx = 0.05 Н (сила вправо)
      action_forward[:, 1] = 0.05  # fx = 0.05 Н (сила вверх)

      for step in range(POS_TEST_STEPS):
          next_state, reward, terminated, truncated, info = env.step(action_forward)

          if step % 10 == 0:
              print(f"Шаг {step}: Позиция блока 0 - x={next_state[0]:.4f}, y={next_state[1]:.4f}, z={next_state[2]:.4f}")

    if ROT_TEST_STEPS != 0:
      print("\n=== Тест вращения блоков ===")

      action_rotate = np.zeros((10, 6), dtype=np.float32)
      action_rotate[:, 5] = 0.005  # tz = 0.005 Н·м (момент вокруг оси Z)

      for step in range(ROT_TEST_STEPS):
          next_state, reward, terminated, truncated, info = env.step(action_rotate)

          # Выводим состояние каждые 10 шагов
          if step % 10 == 0:
              print(f"Шаг {step}: Позиция блока 0 - x={next_state[0]:.4f}, y={next_state[1]:.4f}, z={next_state[2]:.4f}")

    print("\n=== Тест завершен ===")

    # Закрываем окружение
    env.close()
