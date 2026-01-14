from environment import *


n_blocks = 6

random.seed(123)
np.random.seed(123)
env = make_jenga_env(n_blocks=n_blocks, render=True)

one_test_length = 25
z = int(BINS_COUNT / 2)

action_roll = [0, z, z, z, BINS_COUNT - 1, z, z] # spin around x
action_pitch = [0, z, z, z, z, BINS_COUNT - 1, z] # spin around y
action_yaw = [0, z, z, z, z, z, BINS_COUNT - 1] # spin around z

action_x = [0, BINS_COUNT - 1, z, z, z, z, z] # fly x
action_y = [0, z, BINS_COUNT - 1, z, z, z, z] # fly y
action_z = [0, z, z, BINS_COUNT - 1, z, z, z] # fly z


action_arr = [action_roll, action_pitch, action_yaw, action_x, action_y, action_z]

for action in action_arr:
    env.reset()
    for i in range(0, one_test_length):
        z = int(BINS_COUNT / 2)
        env.step(action)
        env.render()
        time.sleep(0.1)
