# %%
import random
from environment import make_jenga_env
from model_trainer import JengaML_Trainer
import numpy as np
import time

# %%
n_blocks = 10

random.seed(123)
np.random.seed(123)
env = make_jenga_env(n_blocks=n_blocks, render=True)
trainer = JengaML_Trainer(env, blocks_count=n_blocks, total_timesteps=1000000, n_steps=5000)

# %%
env.reset()
trainer.train()

trainer.evaluate(True)

# %%
