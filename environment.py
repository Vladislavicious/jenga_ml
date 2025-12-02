# there will be all MuJoCo-related stuff


from typing import List


class FakeEnvironment:
    def __init__(self):
        self._state_dim = 10

    def get_state_dim(self) -> int:
        return self._state_dim

    def get_action_dims(self) -> List[int]:
        return [
            self._state_dim,
            11,
            11,
            11,
        ]  # 10 blocks, 11 values each for power on X, Y, Z
