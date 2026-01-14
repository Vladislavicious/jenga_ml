# %%
import random
from environment import make_jenga_env
from model_trainer import JengaML_Trainer
import numpy as np
import time

# %%
n_blocks = 6

random.seed(123)
np.random.seed(123)
env = make_jenga_env(n_blocks=n_blocks, render=True)

steps_in_iteration = 100
iterations = 500
total_steps = steps_in_iteration * iterations

trainer = JengaML_Trainer(env, blocks_count=n_blocks, total_timesteps=total_steps, n_steps=steps_in_iteration)

# %%
env.reset()
trainer.train()

trainer.evaluate(max_steps=steps_in_iteration, visualize=True)

# %%
