
#%%
from simulation_module import get_full_transition_matrix
import yaml
from munch import munchify
import pickle
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)

rewards = config.params.rewards_set[0]
options_set = config.params.options_set
memory_size = config.params.memory_size_set[0]

if options_set == 'antistereo':
    options_set_fname = "crows_pairs_antistereo_sample.pkl"
    options_dict = pickle.load(open(options_set_fname, 'rb'))
    #options_set = [options_dict[i]['differences'] for i in options_dict.keys()]

if options_set == 'stereo':
    options_set_fname = "crows_pairs_stereo_sample.pkl"
    options_dict = pickle.load(open(options_set_fname, 'rb'))

options_id = list(options_dict.keys())[0]
options = options_dict[options_id]['differences']

if config.sim.fill_blank == False:
    topic = None
else:
    topic = options_dict[options_id]['template']
                
get_full_transition_matrix(options = options, memory_size=memory_size, rewards=rewards, topic = topic, options_id=options_id)