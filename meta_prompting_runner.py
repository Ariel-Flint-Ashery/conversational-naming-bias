#%% imports
import pickle
import simulation_module as sm
import yaml
from munch import munchify

#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%%
rewards_set = config.params.rewards_set
memory_size_set = config.params.memory_size_set
shorthand = config.model.shorthand
options_set = config.params.options_set
minority_size_set = config.minority.minority_size_set

#%%
if options_set == 'antistereo':
    options_set_fname = "crows_pairs_antistereo_sample.pkl"
    options_dict = pickle.load(open(options_set_fname, 'rb'))
    #options_set = [options_dict[i]['differences'] for i in options_dict.keys()]

if options_set == 'stereo':
    options_set_fname = "crows_pairs_stereo_sample.pkl"
    options_dict = pickle.load(open(options_set_fname, 'rb'))

#%%
options = options_dict[list(options_dict.keys())[0]]['differences']
topic = options_dict[list(options_dict.keys())[0]]['template']
#%%
mp_fname = f"data/{shorthand}_meta_test.pkl"
tracker = sm.simulate_meta_prompting(memory_size=memory_size_set[0], rewards=rewards_set[0], options=options, fname = mp_fname, topic = topic)
f = open(mp_fname, 'wb')
pickle.dump(tracker, f)
f.close()
# %%
