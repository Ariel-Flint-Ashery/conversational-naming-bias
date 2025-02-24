# a file to run the entire simulation
#%% imports
import pickle
import simulation_module as sm
import yaml
from munch import munchify
import utils as ut
import prompting as pr
from pathlib import Path
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%%
N = config.params.N
runs = config.params.runs
convergence_time = config.params.convergence_time
rewards_set = config.params.rewards_set
memory_size_set = config.params.memory_size_set
initial_composition = config.params.initial_composition
initial = config.params.initial
total_interactions = config.params.total_interactions
temperature = config.params.temperature
committment_index = config.minority.committment_index
convergence_threshold = config.params.convergence_threshold
shorthand = config.model.shorthand
options_set = config.params.options_set
minority_size_set = config.minority.minority_size_set
network_type = config.network.network_type
version = config.sim.version
initial = config.params.initial
initial_composition = config.params.initial_composition
continue_evolution = config.sim.continue_evolution
#%%
# load options
if options_set == 'antistereo':
    options_set_fname = "crows_pairs_antistereo_sample.pkl"
    options_dict = pickle.load(open(options_set_fname, 'rb'))
    #options_set = [options_dict[i]['differences'] for i in options_dict.keys()]

if options_set == 'stereo':
    options_set_fname = "crows_pairs_stereo_sample.pkl"
    options_dict = pickle.load(open(options_set_fname, 'rb'))
    #options_set = [options_dict[i]['differences'] for i in options_dict.keys()]
#%% 
print(options_set)
def bias_runner():
    for rewards in rewards_set:
        for m in memory_size_set:
            for options_id in options_dict.keys():
                options = options_dict[options_id]['differences']
                topic = options_dict[options_id]['template']

                # INDIVIDUAL TESTS

                mainfname = f"data/{shorthand}_individual_bias_test_ID_{options_id}_{temperature}tmp" + ".pkl"
                print(mainfname)
                try:
                    mainframe = pickle.load(open(mainfname, 'rb'))
                except:
                    mainframe = {'simulation': ut.get_player(), 'tracker': {'answers': []}}
                
                sm.individual(dataframe=mainframe, memory_size=0, rewards = rewards, repeats=1000, options = options, fname = mainfname, topic = topic)
                f = open(mainfname, 'wb')
                pickle.dump(mainframe, f)
                f.close()
                
                # COLLECTIVE TESTS

                # first, we load a baseline model
                mainfname = '.pkl'
                # load a converged baseline
                if initial == 'None':
                    mainfname = f"data/{shorthand}_converged_baseline_ID_{options_id}_{rewards[0]}_{rewards[1]}_{m}mem_{config.network.network_type}_{N}ps_{temperature}tmp.pkl"
                
                else:
                    #raise ValueError("initial must be set to 'None'. Evolution is not supported yet")
                    mainfname = f"data/{shorthand}_evolved_from_{initial}_ID_{options_id}_{rewards[0]}_{rewards[1]}_{m}mem_{config.network.network_type}_{N}ps_{total_interactions}ints_{temperature}tmp.pkl"
                print(mainfname)
                mainframe = ut.load_mainframe(mainfname)
                mainframe['rules'] = pr.get_rules(rewards, options = options, topic = topic)

                # run until sim converges
                for run in range(runs):
                    temp_fname = "temporary_" + mainfname
                    if initial == 'None':
                        if len(mainframe.keys())-1 > run:
                            df = mainframe[run]
                            #continue
                        else:
                            print(f"--- STARTING RUN {run} ---")
                            print("---------- BASELINE CONVERGENCE ----------")
                            df = ut.get_empty_population(fname=temp_fname)
                        sm.population(dataframe=df, run=run, memory_size=m, rewards=rewards, options=options, fname=temp_fname)
                    if initial != 'None':
                        if len(mainframe.keys())-1 > run:
                            continue
                        print(f"--- STARTING RUN {run} ---")
                        df = ut.get_prepared_population(fname=temp_fname, rewards=rewards, options=options, minority_size=0, memory_size=m)
                        print("---------- CONTINUING EVOLUTION ----------")
                        sm.committed(dataframe=df, run=run, memory_size=m, rewards=rewards, options=options, fname=temp_fname, total_interactions=total_interactions)
                    print(run)
                    # save in main dataframe
                    mainframe[run] = df

                    f = open(mainfname, 'wb')
                    pickle.dump(mainframe, f)
                    f.close()

                    # delete temporary file
                    file_to_rem = Path(temp_fname)
                    file_to_rem.unlink(missing_ok=True)

def committed_runner():
    for rewards in rewards_set:
        for memory_size in memory_size_set:
            for run in range(runs):
                for cm in minority_size_set:
                    for options_id in options_dict.keys():
                        options = options_dict[options_id]['differences']
                        topic = options_dict[options_id]['template']
                        if initial != 'None':
                            mainframe = ut.get_prepared_population(fname='.pkl', rewards=rewards, options=options, minority_size=0, memory_size=memory_size)
                        else:
                            raise ValueError("baseline does not exist")
                            
                        cmfname = f"data/{shorthand}_{version}_{options[initial]}_{cm}cmtd_ID_{options_id}_{rewards[0]}_{rewards[1]}_{memory_size}mem_{config.network.network_type}_{N}ps_{temperature}tmp.pkl"
                        print(cmfname)
                        cmframe = ut.load_mainframe(fname=cmfname)
                        temp_fname = "temporary_" + cmfname
                        print("cmframe keys:", cmframe.keys())
                        # check if we already simulated this run
                        if len(cmframe.keys()) > run:
                            df = cmframe[run]
                        # if not, use old dataframe to run convergence.
                        else:
                            # load temporary dataframe

                            df = ut.load_mainframe(fname = temp_fname)

                            # check if temporary dataframe is full.
                            if len(df.keys()) == 0:
                                print(f'----------STARTING RUN {run} FROM SCRATCH----------')
                                df = mainframe
                            
                                # add committed agents to baseline dataframe
                                if version == 'swap':
                                    print("---------- SWAPPING COMMITTED AGENTS ----------")
                                    df = ut.swap_committed(df, cm)
                                
                                if version == 'inject':
                                    print("---------- ADDING COMMITTED AGENTS ----------")
                                    df = ut.add_committed(df, cm)

                            print(f"Run: {run}")
                            print(f"Initial population: {N}")
                            print(f"There are {len(df['simulation'].keys())} players in the game")
                            print(f"minority size: {cm}")
                            word =  df['convergence']['committed_to']
                            print(f'committment word is: {word}')
                            committed_agent_ids = [player for player in df['simulation'].keys() if df['simulation'][player]['committed_tag'] == True]
                            print(f"There are {len(committed_agent_ids)} committed agents: {committed_agent_ids}")
                            # run committed minorities
                            print("---------- RUNNING COMMITTED AGENTS ----------")
                            sm.committed(dataframe=df, run=run, memory_size=memory_size, rewards=rewards, options=options, fname=temp_fname, topic = topic, total_interactions=total_interactions)
                            
                            cmframe[run] = df
                            # save in main dataframe
                            f = open(cmfname, 'wb')
                            pickle.dump(cmframe, f)
                            f.close()
            
                            # delete temporary file
                            file_to_rem = Path(temp_fname)
                            file_to_rem.unlink(missing_ok=True)

#%% RUN

if len(minority_size_set)>0:
    if 0 in minority_size_set:
        print("RUNNING CONVERGENCE/EVOLUTION ONLY --- NO MINORITY")
        bias_runner()
    else:
        committed_runner()

# %%
