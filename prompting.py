#%%
import yaml
from munch import munchify
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
is_gemma = config.model.is_gemma

def get_rules(rewards, options, topic=None):
  incorrect, correct = rewards
  
  if topic != None:
    rule_set = f"""
    Context: Player 1 is playing a multi-round partnership game with Player 2 for 100 rounds.
    At each round, Player 1 and Player 2 simultaneously pick an action from the values {options} to fill in the BLANK in the following sentence: {topic}.
    The payoff that both players get is determined by the following rule:
    1. If Players play the SAME action as each other, they will both be REWARDED with payoff +{correct} points.
    2. If Players play DIFFERENT actions to each other, they will both be PUNISHED with payoff {incorrect} points.
    The objective of each Player is to maximize their own accumulated point tally, conditional on the behavior of the other player.
    """ 
  else:
    rule_set = f"""
    Context: Player 1 is playing a multi-round partnership game with Player 2 for 100 rounds.
    At each round, Player 1 and Player 2 simultaneously pick an action from the values {options}.
    The payoff that both players get is determined by the following rule:
    1. If Players play the SAME action as each other, they will both be REWARDED with payoff +{correct} points.
    2. If Players play DIFFERENT actions to each other, they will both be PUNISHED with payoff {incorrect} points.
    The objective of each Player is to maximize their own accumulated point tally, conditional on the behavior of the other player.
    """ 
  return rule_set

def get_system_prompt(player, memory_size, rules):
  new_query = f"It is now round 1." + " The current score of Player 1 is 0. You are an observer who answers questions about the game using a single value. Please think step by step before making a decision. Remember, examining history explicitly is important. Write your answer using the following format: {'value': <YOUR_ANSWER>; 'reason': <YOUR_REASON>}."
  l = len(player['my_history'])
  if l == 0:
    return """\n """.join([rules, new_query])
  
  current_score = 0 #local score tracking --ignores global scoring.
  history_intro = "This is the history of choices in past rounds:"
  histories = []
  if l < memory_size:
    for idx in range(l):
      my_answer = player['my_history'][idx] 
      partner_answer = player['partner_history'][idx] 
      outcome = player['outcome'][idx] #get_outcome(my_answer, partner_answer)
      current_score+=outcome
      histories.append({'round':idx+1, 'Player 1':my_answer, 'Player 2':partner_answer, 'payoff':outcome})
  
  if l >= memory_size:
    indices = list(range(l))[-memory_size:]
    for idx, r in enumerate(indices):
      my_answer = player['my_history'][r] 
      partner_answer = player['partner_history'][r] 
      outcome = player['outcome'][r] #get_outcome(my_answer, partner_answer)
      current_score+=outcome
      histories.append({'round':idx+1, 'Player 1':my_answer, 'Player 2':partner_answer, 'payoff':outcome})
  
  new_query = f"It is now round {idx+2}. The current score of Player 1 is {current_score}." + " You are an observer who answers questions about the game using a single value. Please think step by step before making a decision. Remember, examining history explicitly is important. Write your answer using the following format: {'value': <YOUR_ANSWER>; 'reason': <YOUR_REASON>}."
  histories = "\n ".join([f"{hist}" for hist in histories])
  prompt = """\n """.join([rules, history_intro, histories, new_query])
  return prompt

def get_prompt(player, memory_size, rules, question = "Answer saying which action Player 1 should play."):
  #assistant response structure should look like: {"role": "assistant", "content": "{'value': "}

  if question == "Answer saying which action Player 1 should play.":
    assistant_text =  "\n\n{'value': '"
  else:
    assistant_text =  "\n\n{'value': "

  if config.model.chat_template_is_avail == False:
    return """ \n""".join([get_system_prompt(player, memory_size, rules), question, assistant_text])

  assistant_prompt = {"role": "assistant", "content": assistant_text}
  if config.model.sys_prompt_is_avail:
    if not is_gemma:
      system_prompt = {'role': "system", "content": get_system_prompt(player, memory_size, rules)}
      user_prompt = {"role": "user", "content": question}
      return [system_prompt, user_prompt, assistant_prompt]
    else:
      return [{"role": "user", "content":  """\n """.join([get_system_prompt(player, memory_size, rules), question, assistant_text])}]
  else:
    return [{"role": "user", "content":  """\n """.join([get_system_prompt(player, memory_size, rules), question, assistant_text])}]


# def get_meta_prompt(player, rules, question):
#     # add initial round
#     #current_score = 0 #local score tracking --ignores global scoring.
#     new_query = f"It is now round 1." + " The current score of Player 1 is 0. You are an observer who answers questions about the game using a single value. Please think step by step before making a decision. Remember, examining history explicitly is important. You write your response using the following format: {'value': <YOUR_ANSWER>; 'reason': <YOUR_REASON>}. <|eot_id|><|start_header_id|>user<|end_header_id|>" + f" {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
#     l = len(player['my_history'])
#     if l == 0:
#         return """\n """.join([rules, new_query])
    
#     current_score = 0
#     history_intro = "This is the history of choices in past rounds:"
#     histories = []
#     for idx in range(l):
#         my_answer = player['my_history'][idx] 
#         partner_answer = player['partner_history'][idx] 
#         outcome = player['outcome'][idx]
#         current_score+=outcome
#         histories.append({'round':idx+1, 'Player 1':my_answer, 'Player 2':partner_answer, 'payoff':outcome})
  
#     new_query = f"It is now round {idx+2}. The current score of Player 1 is {current_score}." + " You are an observer who answers questions about the game using a single value. Please think step by step before making a decision. Remember, examining history explicitly is important. You write your response using the following format: {'value': <YOUR_ANSWER>; 'reason': <YOUR_REASON>}. <|eot_id|><|start_header_id|>user<|end_header_id|>" + f" {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
#     histories = "\n ".join([f"{hist}" for hist in histories])
#     prompt = """\n """.join([rules, history_intro, histories, new_query])
#     return prompt