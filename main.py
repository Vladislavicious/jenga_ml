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

step_count = 1500
iterations = 1500
total_steps = step_count * iterations

trainer = JengaML_Trainer(env, blocks_count=n_blocks, total_timesteps=total_steps, n_steps=step_count)

# %%
env.reset()
trainer.train()

trainer.evaluate(max_steps=step_count, visualize=True)

# %%
