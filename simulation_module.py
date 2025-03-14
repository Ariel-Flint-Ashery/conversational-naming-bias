#%% imports
import random
import prompting as pr
import pickle
import yaml
from munch import munchify
import utils as ut
import meta_prompting as mp
import sqlite3
import json
import time
from itertools import permutations
from scipy.spatial.distance import jensenshannon
import numpy as np
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%% constants
total_interactions = config.params.total_interactions
N = config.params.N
#%% load running functions
if config.sim.mode == 'api':
    import run_API as ask
if config.sim.mode == 'gpu':
    import run_local as ask

import re

def extract_options_id(s):
    match = re.search(r'ID_(\d+)_', s)
    return int(match.group(1)) if match else None

#%% meta prompting
# def simulate_meta_prompting(memory_size, rewards, options, fname, topic):
#     question_list = ['min', 'max', 'actions', 'payoff', 'round', 'action_i', 'points_i', 'no_actions', 'no_points']
#     try:
#         tracker = pickle.load(open(fname, 'rb'))
#     except:
#         tracker = {q: {'responses': [], 'outcome': []} for q in question_list}
#     # choose random player
#     new_options = options.copy()
#     # load their current history up to given round.
#     while len(tracker[question_list[0]]['outcome'])<100:
#         t = len(tracker[question_list[0]]['outcome'])
#         random.shuffle(new_options)
#         rules = pr.get_rules(rewards, options = new_options, topic = topic)

#         running_player = mp.running_player(options = new_options, memory_size=memory_size, rewards=rewards)
#         # get questions
#         i, questions, q_list, prompts = mp.get_meta_prompt_list(some_player = running_player, rules=rules, options=new_options)

#         # get answers
#         # responses = []
#         # gold_responses = []
#         for prompt, question, q in zip(prompts, questions, q_list):
#             print(f"QUESTION: {question}", flush = True)
#             #print(prompt)
#             response = ask.get_meta_response(prompt)
#             gold_response = mp.gold_sim(q, question, running_player, i, options)
#             tracker[q]['responses'].append(response)
#             if q == 'actions':
#                 if all(option in response for option in options):
#                     tracker[q]['outcome'].append(1)
#                     print('SUCCESS', flush = True)
#                 else:
#                     tracker[q]['outcome'].append(0)
#             else:
#                 print(f"GOLD: {gold_response}", flush = True) 
#                 if gold_response in response:
#                     tracker[q]['outcome'].append(1)
#                     print('SUCCESS', flush = True)
#                 else:
#                     tracker[q]['outcome'].append(0)
#             #time.sleep(2)
#         print(f"INTERACTION {t}", flush = True)
#         #if t % 5 == 0:
#         f = open(fname, 'wb')
#         pickle.dump(tracker, f)
#         f.close()
#     return tracker

#%% collective bias testing
def population(dataframe, run, memory_size, rewards, options, fname, topic, initial):
    options_id = extract_options_id(fname)
    new_options = options.copy()
    interaction_dict = dataframe['simulation']
    tracker = dataframe['tracker']

    first_target_id_dict = ask.encode_decode_options(options = options)
    # get empty memory transitions

    all_options = list(permutations(options))
    for opts in all_options:
        rules = pr.get_rules(rewards, options = opts, topic = topic)
        # get prompt with rules & history of play
        prompt = pr.get_prompt(player=ut.get_player(), memory_size=0, rules = rules)

        # get agent response
        matrix_fname = f"matrices/TRANSITION_MATRIX_{config.model.shorthand}_ID_{options_id}_{'_'.join([str(r) for r in rewards])}_{config.params.temperature}tmp_BLANK_{config.sim.fill_blank}.db"
        matrix = get_transition_matrix(fname = matrix_fname, options = opts, my_history=[], partner_history=[], prompt = prompt, first_target_id_dict=first_target_id_dict)


    while ut.has_tracker_converged(tracker) == False:
        if len(tracker['outcome']) > 4000:
            break
        # randomly choose player and a neighbour
        p1 = random.choice(list(interaction_dict.keys()))
        p2 = random.choice(interaction_dict[p1]['neighbours'])
        
        # add interactions to play history
        
        interaction_dict[p1]['interactions'].append(p2)
        interaction_dict[p2]['interactions'].append(p1)
        p1_dict = interaction_dict[p1]
        p2_dict = interaction_dict[p2]
        
        # play

        answers = []
        player_dicts = [p1_dict, p2_dict]
        for idx, player in enumerate(player_dicts):
            random.shuffle(new_options)
            if isinstance(initial, float) and len(player['my_history']) == 0:
                answers.append(player['first_choice'])
                print('first choice')
            elif isinstance(initial, int) and len(tracker['outcome'])==0 and idx == 0 and config.sim.consensus_evolution == True:
                print(f'FIRST PERTURBATION: {options[initial-1]}')
                answers.append(options[initial-1])

            rules = pr.get_rules(rewards, options = new_options, topic = topic)
            # get prompt with rules & history of play
            prompt = pr.get_prompt(player, memory_size=memory_size, rules = rules)

            # use transition matrix
            matrix_fname = f"matrices/TRANSITION_MATRIX_{config.model.shorthand}_ID_{options_id}_{'_'.join([str(r) for r in rewards])}_{config.params.temperature}tmp_BLANK_{config.sim.fill_blank}.db"
            matrix = get_transition_matrix(fname = matrix_fname, options = new_options, my_history=player['my_history'][-memory_size:], partner_history=player['partner_history'][-memory_size:], prompt = prompt, first_target_id_dict = first_target_id_dict)

            # get agent response
            answers.append(sample_from_dict(transition_matrix=matrix, options = new_options))
            #answers.append(ask.get_response(prompt, options=new_options))
                
        my_answer, partner_answer = answers

        # calculate outcome and update dictionary
        
        outcome = ut.get_outcome(my_answer, partner_answer, rewards)
        interaction_dict[p1] = ut.update_dict(p1_dict, my_answer, partner_answer, outcome)
        interaction_dict[p2] = ut.update_dict(p2_dict, partner_answer, my_answer, outcome)
        ut.update_tracker(tracker, p1, p2, my_answer, partner_answer, outcome)
        
        if len(tracker['outcome']) % 50 == 0:
            print(f"CONVERGENCE RUN {run} -- INTERACTION {len(tracker['outcome'])}")
            dataframe['simulation'] = interaction_dict
            dataframe['tracker'] = tracker
            f = open(fname, 'wb')
            pickle.dump(dataframe, f)
            f.close()

    dataframe['simulation'] = interaction_dict
    dataframe['tracker'] = tracker
    if ut.has_tracker_converged(tracker):
        dataframe['convergence'] = {'converged_index': len(tracker['outcome']), 'committed_to': None}
    else:
        dataframe['convergence'] = {'converged_index': None, 'committed_to': None}

#%% COMMITTED MINORITY

def committed(dataframe, run, memory_size, rewards, options, fname, topic, initial,
              total_interactions = total_interactions):
    options_id = extract_options_id(fname)
    new_options = options.copy()
    interaction_dict = dataframe['simulation']
    tracker = dataframe['tracker']
    init_tracker_len = dataframe['convergence']['converged_index']
    first_target_id_dict = ask.encode_decode_options(options = options)
    while len(tracker['outcome']) - init_tracker_len < total_interactions:
        rules = pr.get_rules(rewards, options = new_options, topic = topic)

        # randomly choose player and a neighbour
        p1 = random.choice(list(interaction_dict.keys()))
        p2 = random.choice(interaction_dict[p1]['neighbours'])
        
        # add interactions to play history
        
        interaction_dict[p1]['interactions'].append(p2)
        interaction_dict[p2]['interactions'].append(p1)
        p1_dict = interaction_dict[p1]
        p2_dict = interaction_dict[p2]
        
        # play

        answers = []
        for player in [p1_dict, p2_dict]:
            random.shuffle(new_options)

            # check if committed. If True, play committed answer.
            if player['committed_tag'] == True:
                a = dataframe['convergence']['committed_to']
                answers.append(a)

            elif type(initial) == float and len(player['my_history']) == 0:
                answers.append(player['first_choice'])
                
            else:
                rules = pr.get_rules(rewards, options = new_options, topic = topic)
                # get prompt with rules & history of play
                prompt = pr.get_prompt(player, memory_size=memory_size, rules = rules)

                # use transition matrix
                matrix_fname = f"matrices/TRANSITION_MATRIX_{config.model.shorthand}_ID_{options_id}_{'_'.join([str(r) for r in rewards])}_{config.params.temperature}tmp_BLANK_{config.sim.fill_blank}.db"
                matrix = get_transition_matrix(fname = matrix_fname, options = new_options, my_history=player['my_history'][-memory_size:], partner_history=player['partner_history'][-memory_size:], prompt = prompt, first_target_id_dict = first_target_id_dict)

                # get agent response
                answers.append(sample_from_dict(transition_matrix=matrix, options = new_options))
                
        my_answer, partner_answer = answers

        # calculate outcome and update dictionary
        
        outcome = ut.get_outcome(my_answer, partner_answer, rewards)
        interaction_dict[p1] = ut.update_dict(p1_dict, my_answer, partner_answer, outcome)
        interaction_dict[p2] = ut.update_dict(p2_dict, partner_answer, my_answer, outcome)
        ut.update_tracker(tracker, p1, p2, my_answer, partner_answer, outcome)
        
        if len(tracker['outcome']) % 20 == 0:
            print(fname)
            print(f"PREPARED RUN {run} -- INTERACTION {len(tracker['outcome'])}")
            dataframe['simulation'] = interaction_dict
            dataframe['tracker'] = tracker
            f = open(fname, 'wb')
            pickle.dump(dataframe, f)
            f.close()


    dataframe['simulation'] = interaction_dict
    dataframe['tracker'] = tracker
# %%
def sample_from_dict(transition_matrix, options):

    probability_vector = [transition_matrix[option] for option in options]
    return options[ut.log_roulette_wheel(probability_vector)]
# def get_matrix_fname(stable_options, multiplier, rewards):
#     matrix_fname = f"TRANSITION_MATRIX_{stable_options}_MULTI_{multiplier}_{'_'.join([str(r) for r in rewards])}_{config.params.temperature}tmp.pkl"
#     return matrix_fname
def get_transition_matrix(fname, options, my_history, partner_history, prompt, first_target_id_dict):
    """
    Loads or computes a transition probability dictionary and saves it to SQLite for concurrent access.
    """

    memory_string = '_'.join(my_history) + '_&_' + '_'.join(partner_history)
    options_string = '_'.join(options)

    # Connect to SQLite database
    conn = sqlite3.connect(fname)
    conn.execute("PRAGMA journal_mode=WAL;")  # Enable concurrent reads while writing
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transition_matrix (
            memory_string TEXT,
            options_string TEXT,
            probability_dict TEXT,
            PRIMARY KEY (memory_string, options_string)
        )
    """)

    # Check if the entry exists
    cursor.execute("SELECT probability_dict FROM transition_matrix WHERE memory_string = ? AND options_string = ?", 
                   (memory_string, options_string))
    row = cursor.fetchone()

    if row:
        # Load existing probability_dict
        probability_dict = json.loads(row[0])
    else:
        # Compute new transition probabilities
        probability_dict = ask.get_probability_dict(options=options, prompt=prompt, first_target_id_dict=first_target_id_dict)

        # Save to database
        cursor.execute("INSERT OR REPLACE INTO transition_matrix (memory_string, options_string, probability_dict) VALUES (?, ?, ?)", 
                       (memory_string, options_string, json.dumps(probability_dict)))

        conn.commit()

    conn.close()
    #print(probability_dict)
    #time.sleep(5)
    return probability_dict

def get_full_transition_matrix(options, memory_size, rewards, options_id, topic = None):
    first_target_id_dict = ask.encode_decode_options(options = options)
    # get empty memory transitions
    action_vectors = ut.generate_action_vectors(options = options, memory_size = memory_size)
    all_options = list(permutations(options))
    for opts in all_options:
        #print(opts)
        for pair in action_vectors:
            print(pair)
            player=ut.get_player()
            m = len(pair[0])
            for h in range(m):
                my_answer, partner_answer = [p[h] for p in pair]
                ut.update_dict(player, my_answer, partner_answer, ut.get_outcome(my_answer,partner_answer, rewards))

            rules = pr.get_rules(rewards, options = opts, topic = topic)
            # get prompt with rules & history of play
            prompt = pr.get_prompt(player = player, memory_size=m, rules = rules)
            # get agent response
            matrix_fname = f"matrices/TRANSITION_MATRIX_{config.model.shorthand}_ID_{options_id}_{'_'.join([str(r) for r in rewards])}_{config.params.temperature}tmp_BLANK_{config.sim.fill_blank}.db"
            matrix = get_transition_matrix(fname = matrix_fname, options = opts, my_history=player['my_history'], partner_history=player['partner_history'], prompt = prompt, first_target_id_dict=first_target_id_dict)

# def find_balanced_pair(rewards, epsilon = 0.001):
#     fname = f"data/{config.model.shorthand}_random_option_pairs_{epsilon}.pkl"
#     try:
#         all_pairs = pickle.load(open(fname, 'rb'))
#     except:
#         all_pairs = []
#     all_probs = []
#     while len(all_pairs)<10:
#         pair = ut.generate_unique_strings(n=2, k=6)
#         first_target_id_dict = ask.encode_decode_options(options = pair)

#         if len(set(list(first_target_id_dict.values()))) == 1:
#             continue
#         # get empty memory transitions
        
#         all_options = list(permutations(pair))
#         pair_dict = {p: [] for p in pair}
#         for opts in all_options:
#             #print(opts)
#             rules = pr.get_rules(rewards, options = opts, topic = topic)
#             # get prompt with rules & history of play
#             prompt = pr.get_prompt(player=ut.get_player(), memory_size=0, rules = rules)
#             # get agent response
#             probability_dict = ask.get_probability_dict(options=opts, prompt=prompt, first_target_id_dict=first_target_id_dict)
#             for p in pair:
#                 pair_dict[p].append(probability_dict[p])
            
#         probs = [sum(np.exp(pair_dict[p]))/2 for p in pair]

#         if jensenshannon([0.5, 0.5], probs) < epsilon:
#             print(f"FOUND CANDIDATE: {pair}, PROB = {probs}")
#             all_pairs.append(pair)
#             all_probs.append(probs)
#             f = open(fname, 'wb')
#             pickle.dump(all_pairs, f)
#             f.close()

    
#     # save data

#     print(all_pairs)
#     print(all_probs)
#     f = open(fname, 'wb')
#     pickle.dump(all_pairs, f)
#     f.close()

#     return all_pairs
# %%

# LEGACY CODE

# #%% imports
# import random
# import prompting as pr
# import pickle
# import yaml
# from munch import munchify
# import utils as ut
# import meta_prompting as mp
# #%%
# with open("config.yaml", "r") as f:
#     doc = yaml.safe_load(f)
# config = munchify(doc)
# #%% constants
# total_interactions = config.params.total_interactions
# N = config.params.N
# #%% load running functions
# if config.sim.mode == 'api':
#     import run_API as ask
# if config.sim.mode == 'gpu':
#     import run_local as ask

# #%% meta prompting
# def simulate_meta_prompting(memory_size, rewards, options, fname, topic):
#     question_list = ['min', 'max', 'actions', 'payoff', 'round', 'action_i', 'points_i', 'no_actions', 'no_points']
#     try:
#         tracker = pickle.load(open(fname, 'rb'))
#     except:
#         tracker = {q: [] for q in question_list}
#     # choose random player
#     new_options = options.copy()
#     # load their current history up to given round.
#     while len(tracker[question_list[0]])<100:
#         t = len(tracker[question_list[0]])
#         random.shuffle(new_options)
#         rules = pr.get_rules(rewards, options = new_options, topic=topic)

#         running_player = mp.running_player(options = new_options, memory_size=memory_size, rewards=rewards)
#         # get questions
#         i, questions, q_list, prompts = mp.get_meta_prompt_list(some_player = running_player, rules=rules, options=new_options)

#         # get answers
#         # responses = []
#         # gold_responses = []
#         for prompt, question, q in zip(prompts, questions, q_list):
#             #print(question)
#             #print(prompt)
#             response = ask.get_meta_response(prompt)
#             gold_response = mp.gold_sim(q, question, running_player, i, options)
            
#             if q == 'actions':
#                 if all(option in response for option in options):
#                     tracker[q].append(1)
#                     print('Success')
#                 else:
#                     tracker[q].append(0)
#             else:
#                 print("GOLD: ", gold_response) 
#                 if gold_response in response:
#                     tracker[q].append(1)
#                     print('SUCCESS')
#                 else:
#                     tracker[q].append(0)
#             #time.sleep(2)
#         print(f"INTERACTION {t}")
#         if t % 5 == 0:
#             f = open(fname, 'wb')
#             pickle.dump(tracker, f)
#             f.close()
#     return tracker

# #%% individual bias testing
# def individual(dataframe, memory_size, rewards, options, fname, repeats, topic):
#     new_options = options.copy()
#     player = dataframe['simulation']
#     tracker = dataframe['tracker']
#     while len(tracker['answers']) < repeats:
#         random.shuffle(new_options)
#         rules = pr.get_rules(rewards, options = new_options, topic=topic)
        
#         # play
#         # get prompt with rules & history of play
#         prompt = pr.get_prompt(player, memory_size=memory_size, rules = rules)

#         # get agent response
#         answer = ask.get_response(prompt, options=new_options)

#         tracker['answers'].append(answer)
        
#         if len(tracker['answers']) % 50 == 0:
#             print(f"INTERACTION {len(tracker['answers'])}")
#             dataframe['tracker'] = tracker
#             f = open(fname, 'wb')
#             pickle.dump(dataframe, f)
#             f.close()

#     dataframe['tracker'] = tracker

# #%% collective bias testing
# def population(dataframe, run, memory_size, rewards, options, fname, topic):
#     new_options = options.copy()
#     interaction_dict = dataframe['simulation']
#     tracker = dataframe['tracker']
#     while ut.has_tracker_converged(tracker) == False:
#         # randomly choose player and a neighbour
#         p1 = random.choice(list(interaction_dict.keys()))
#         p2 = random.choice(interaction_dict[p1]['neighbours'])
        
#         # add interactions to play history
        
#         interaction_dict[p1]['interactions'].append(p2)
#         interaction_dict[p2]['interactions'].append(p1)
#         p1_dict = interaction_dict[p1]
#         p2_dict = interaction_dict[p2]
        
#         # play

#         answers = []
#         for player in [p1_dict, p2_dict]:
#             random.shuffle(new_options)
#             rules = pr.get_rules(rewards, options = new_options, topic=topic)
#             # get prompt with rules & history of play
#             prompt = pr.get_prompt(player, memory_size=memory_size, rules = rules)

#             # get agent response
#             answers.append(ask.get_response(prompt, options=new_options))
                
#         my_answer, partner_answer = answers

#         # calculate outcome and update dictionary
        
#         outcome = ut.get_outcome(my_answer, partner_answer, rewards)
#         interaction_dict[p1] = ut.update_dict(p1_dict, my_answer, partner_answer, outcome)
#         interaction_dict[p2] = ut.update_dict(p2_dict, partner_answer, my_answer, outcome)
#         ut.update_tracker(tracker, p1, p2, my_answer, partner_answer, outcome)
        
#         if len(tracker['outcome']) % 50 == 0:
#             print(f"RUN {run} -- INTERACTION {len(tracker['outcome'])}")
#             dataframe['simulation'] = interaction_dict
#             dataframe['tracker'] = tracker
#             f = open(fname, 'wb')
#             pickle.dump(dataframe, f)
#             f.close()

#     dataframe['simulation'] = interaction_dict
#     dataframe['tracker'] = tracker
#     dataframe['convergence'] = {'converged_index': len(tracker['outcome']), 'committed_to': None}

# #%% COMMITTED MINORITY

# def committed(dataframe, run, memory_size, rewards, options, fname, topic, total_interactions = total_interactions):
#     new_options = options.copy()
#     interaction_dict = dataframe['simulation']
#     tracker = dataframe['tracker']
#     init_tracker_len = dataframe['convergence']['converged_index']
#     while len(tracker['outcome']) - init_tracker_len < total_interactions:
#         random.shuffle(new_options)
#         rules = pr.get_rules(rewards, options = new_options, topic=topic)

#         # randomly choose player and a neighbour
#         p1 = random.choice(list(interaction_dict.keys()))
#         p2 = random.choice(interaction_dict[p1]['neighbours'])
        
#         # add interactions to play history
        
#         interaction_dict[p1]['interactions'].append(p2)
#         interaction_dict[p2]['interactions'].append(p1)
#         p1_dict = interaction_dict[p1]
#         p2_dict = interaction_dict[p2]
        
#         # play

#         answers = []
#         for player in [p1_dict, p2_dict]:
#             # check if committed. If True, play committed answer.
#             if player['committed_tag'] == True:
#                 a = dataframe['convergence']['committed_to']
#                 answers.append(a)
#             else:
#                 # get prompt with rules & history of play
#                 prompt = pr.get_prompt(player, memory_size=memory_size, rules = rules)

#                 # get agent response
#                 answers.append(ask.get_response(prompt, options=new_options))
                
#         my_answer, partner_answer = answers

#         # calculate outcome and update dictionary
        
#         outcome = ut.get_outcome(my_answer, partner_answer, rewards)
#         interaction_dict[p1] = ut.update_dict(p1_dict, my_answer, partner_answer, outcome)
#         interaction_dict[p2] = ut.update_dict(p2_dict, partner_answer, my_answer, outcome)
#         ut.update_tracker(tracker, p1, p2, my_answer, partner_answer, outcome)
        
#         if len(tracker['outcome']) % 20 == 0:
#             print(fname)
#             print(f"COMMITTED RUN {run} -- INTERACTION {len(tracker['outcome'])}")
#             dataframe['simulation'] = interaction_dict
#             dataframe['tracker'] = tracker
#             f = open(fname, 'wb')
#             pickle.dump(dataframe, f)
#             f.close()


#     dataframe['simulation'] = interaction_dict
#     dataframe['tracker'] = tracker
# # %%
