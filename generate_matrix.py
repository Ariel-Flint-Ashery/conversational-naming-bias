
#%%
from simulation_module import get_full_transition_matrix
import yaml
from munch import munchify
import time
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)

rewards = config.params.rewards_set[0]
options = config.params.options_set[0]
memory_size = config.params.memory_size_set[0]
print(options)
get_full_transition_matrix(options = options, memory_size=memory_size, rewards=rewards)