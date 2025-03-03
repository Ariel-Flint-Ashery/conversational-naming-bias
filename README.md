# conversational-naming-bias
Pragmatic semantic syntactic bias in conversational examples of the experimental naming game framework

# key information before starting:
1. Make sure you create a 'matrices' directory in the same directory as your .py files.
2. Add the sub-directories /spontaneous, /first_step_evolution, /consensus_evolution into your 'data' and 'temporary_data' directories, which should also be in the same place you store the code.
3. committed minority runs are not supported as of yet.
4. Before running a model, check if it supports a chat-based conversation, and if it has a system prompt. Then, select the appropriate config options.

# config information:
1. initial_set: 'None'=regular convergence run starting from prior bias, 0 or 1 = all members start with options[0 or 1]. Float: 0.7 means 70% start with option 1.
2. options_set: For data storage reasons, the running file only supports 'stereo' or 'antistereo', which are saved as .pkl files. You can modify these files in the Jupyter notebook "get_crows_pairs_sample.ipynb".
3. 'consensus_evolution': This is only relevant if you are not using the prior initial conditions. If True, then all agents start with a full memory, with composition dictated by 'initial_set' (initial_set can only take integer values for this option!). If False, the first choice of each agent is fixed, with population composition dictated by 'initial_set'.

Final note: For consensus_evolution tasks, the simulation currently perturbs one agent in very first interaction in the game, selecting the non-consensus option. This is to speed up simulations, but you can comment out the relevant lines of code in the 'simulation_module.py' file.
   
