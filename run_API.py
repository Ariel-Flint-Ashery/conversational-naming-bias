#%%
import requests
import time
import yaml
from munch import munchify
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import numpy as np
import utils as ut
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
# set temperature to 0 for deterministic outcomes
temperature = config.params.temperature
if temperature == 0:
    llm_params = {#"do_sample": False,
            "max_tokens": 8,
            'logprobs': True,
            'top_logprobs': 4
            }
else:
    llm_params = {#"do_sample": True,
            "temperature": temperature,
            #"top_k": 10,
            "max_tokens": 8,
            'logprobs': True,
            'top_logprobs': 4
            }
#%%
API_TOKEN = config.model.API_TOKEN   
headers = {"Authorization": f"Bearer {API_TOKEN}", "x-use-cache": 'false'}
API_URL = "https://api-inference.huggingface.co/models/"+config.model.model_name
tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, token=API_TOKEN)
client = InferenceClient(api_key=API_TOKEN, headers = headers)
#%%
# def query(payload):
#     "Query the Hugging Face API"
#     try:
#         response = requests.post(API_URL, headers=headers, json=payload).json()
#     except:
#         return None
#     return response

def query(payload):
    try:
        response = client.chat.completions.create(
                    model=config.model.model_name, 
                    messages=payload,
                    #response_format={'type': 'json'},
                    **llm_params
                ).choices[0]#.message.content
    except:
        return None
    return response

def API_hit(chat, options, first_target_phrase):
    """Generate a response from the model."""

    overloaded = 1
    while overloaded == 1:
        response = query(chat)#query({"inputs": chat, "parameters": llm_params, "options": {"use_cache": False}})
        #print(response)
        if response == None:
            print('CAUGHT JSON ERROR')
            continue
            #print(chat)
            #raise ValueError

        if type(response)==dict:
            print("AN EXCEPTION: ", response)
            time.sleep(2.5)
            if "Inference Endpoints" in response['error']:
              print("HOURLY RATE LIMIT REACHED")
              time.sleep(450)
            continue
        
        try:
            outputs = [response.logprobs.content[i].top_logprobs for i in range(len(response.logprobs.content))]
            token_outputs = [[o.token for o in output] for output in outputs]
        except:
            continue
        
        if len([i for i in range(len(token_outputs)) if any(phrase==token_outputs[i][0] for phrase in first_target_phrase)]) != 0:
            overloaded = 0
        else:
            print("FIRST PHRASE NOT FOUND IN index 0 POSITION IN ANY TOKEN POSITION")
            print(f"Response: {response.message.content}")
            print("Output tokens:")
            for o in token_outputs:
                print(o, first_target_phrase, any(phrase==o for phrase in first_target_phrase))

        # if any(option in response.message.content for option in options):
        #     overloaded=0

    return [response, outputs, token_outputs]

def get_response(chat, options, first_target_phrase):
    response = API_hit(chat, options, first_target_phrase)[0]
    response_split = response.message.content.split("'")
    for opt in options:
        try:
            index = response_split.index(opt)
        except:
            continue
    #print(response_split[index])
    return response_split[index]

def get_meta_response(chat):
    """Generate a response from the Llama model."""

    overloaded = 1
    while overloaded == 1:
        response = query(chat)#query({"inputs": chat, "parameters": llm_params, "options": {"use_cache": False}})
        #print(response)
        if response == None:
            print('CAUGHT JSON ERROR')
            continue

        if type(response)==dict:
            print("AN EXCEPTION")
            time.sleep(2.5)
            if "Inference Endpoints" in response['error']:
              print("HOURLY RATE LIMIT REACHED")
              time.sleep(900)
        
        
        elif len(response.message.content.split(";")) < 2:
            print(f"RESPONSE SPLIT: {response.message.content.split(';')}")
            overloaded = 1
        # if 'value' in response['generated_text']:
        #     overloaded=0
    
        #     response_split = response['generated_text'].split(";")
        #     response_split = response_split[0].split(": ")
        #     if len(response_split)<2:
        #         overloaded = 1
    response_split = response.message.content.split(";")      
    print(response_split[0])
    #time.sleep(5)
    return response_split[0]

def encode_decode_options(options):
    target_phrase = [[tokenizer.decode(target_token_id, skip_special_tokens=True) for target_token_id in tokenizer.encode(option, add_special_tokens=False)] for option in options]
    #target_phrase = [tokenizer.decode(target_id, skip_special_tokens=True) for target_id in target_ids]
    first_target_phrase = [target[0] for target in target_phrase]
    print(f"Options tokens: {target_phrase}")
    print(f"first options tokens: {first_target_phrase}")
    first_target_id_dict = {option: first_target_phrase[i] for i, option in enumerate(options)}
    return first_target_id_dict

def get_probability_dict(options, prompt, first_target_id_dict, temperature = config.params.temperature, epsilon = 0.000001):
    
    first_target_phrase = [first_target_id_dict[option] for option in options]
    response, outputs, token_outputs = API_hit(chat=prompt, options = options, first_target_phrase = first_target_phrase)
    probability_dict = {opt: -np.inf for opt in options}

    # find the token location where both options exist.
    ## find indices of these locations in each successive generation
    outputs = [response.logprobs.content[i].top_logprobs for i in range(len(response.logprobs.content))]
    token_outputs = [[o.token for o in output] for output in outputs]
    index_list = [i for i in range(len(token_outputs)) if all(phrase in token_outputs[i] for phrase in first_target_phrase)]

    # print(f"Response: {response}")
    # print("Token outputs:")
    # for o in token_outputs:
    #     print(o)
    # now, we have the position of the token - let us take the probability!

    if len(index_list) == 0:
        index = [i for i in range(len(token_outputs)) if any(phrase==token_outputs[i][0] for phrase in first_target_phrase)][0]
        # find the winning word.
        selected_option = options[[idx for idx in range(len(first_target_phrase)) if token_outputs[index][0] in first_target_phrase[idx]][0]]
        winning_prob = 0.0 #response.logprobs.content[index].top_logprobs[0].logprob
        probability_dict[selected_option] = winning_prob
        #return probability_dict
    
    else:
        for i, phrase in enumerate(first_target_phrase):
            # find index of option in vector
            try:
                index = token_outputs[index_list[0]].index(phrase)
            except:
                continue
            
            # find logprob
            selected_option = options[i]
            winning_prob = response.logprobs.content[index_list[0]].top_logprobs[index].logprob
            #print(response.logprobs.content[index_list[0]].top_logprobs[index].token,  np.exp(winning_prob))

            if np.exp(winning_prob) < epsilon:
                winning_prob = -np.inf #np.log(epsilon)
            probability_dict[selected_option] = winning_prob

    # renormalize probability:
    options_log_probs = list(probability_dict.values())
    #print(index_list)

    if -np.inf in options_log_probs:
        # print("Token outputs:")
        # for o in token_outputs:
        #     print(o)
        # print(index_list)
        # print(options_log_probs, np.exp(options_log_probs))
        normed_probs = ut.normalize_probs(np.exp(options_log_probs))
        normed_log_probs = np.log(normed_probs)
        print(normed_log_probs)
        #time.sleep(5)
    else:
        normed_log_probs = ut.normalize_logprobs(options_log_probs)

    for option, log_prob in zip(probability_dict.keys(), normed_log_probs):
        probability_dict[option] = log_prob
    #print(probability_dict)
    return probability_dict

# #%%
# import requests
# import time
# import yaml
# from munch import munchify
# from huggingface_hub import InferenceClient

# #%%
# with open("config.yaml", "r") as f:
#     doc = yaml.safe_load(f)
# config = munchify(doc)
# # set temperature to 0 for deterministic outcomes
# temperature = config.params.temperature
# if temperature == 0:
#     llm_params = {#"do_sample": False,
#             "max_tokens": 12,
#             #"return_full_text": False, 
#             }
# else:
#     llm_params = {#"do_sample": True,
#             "temperature": temperature,
#             #"top_k": 10,
#             "max_tokens": 15,
#             #"return_full_text": False, 
#             }  
# #%%
# API_TOKEN = config.model.API_TOKEN   
# headers = {"Authorization": f"Bearer {API_TOKEN}", "x-use-cache": 'false'}
# API_URL = "https://api-inference.huggingface.co/models/"+config.model.model_name

# client = InferenceClient(api_key=API_TOKEN, headers = headers)
# #%%
# # def query(payload):
# #     "Query the Hugging Face API"
# #     try:
# #         response = requests.post(API_URL, headers=headers, json=payload).json()
# #     except:
# #         return None
# #     return response

# def query(payload):
#     try:
#         response = client.chat.completions.create(
#                     model=config.model.model_name, 
#                     messages=payload,
#                     #response_format={'type': 'json'},
#                     **llm_params
#                 ).choices[0].message.content
#     except:
#         return None
#     return response
# def get_response(chat, options):
#     """Generate a response from the model."""

#     overloaded = 1
#     while overloaded == 1:
#         response = query(chat)#query({"inputs": chat, "parameters": llm_params, "options": {"use_cache": False}})
#         #print(response)
#         if response == None:
#             print('CAUGHT JSON ERROR')
#             continue

#         if type(response)==dict:
#             print("AN EXCEPTION: ", response)
#             time.sleep(2.5)
#             if "Inference Endpoints" in response['error']:
#               print("HOURLY RATE LIMIT REACHED")
#               time.sleep(450)
                
#         elif any(option in response.split("'") for option in options):
#             overloaded=0
#     response_split = response.split("'")
#     for opt in options:
#         try:
#             index = response_split.index(opt)
#         except:
#             continue
#     #print(response_split[index])
#     return response_split[index]

# def get_meta_response(chat):
#     """Generate a response from the Llama model."""

#     overloaded = 1
#     while overloaded == 1:
#         response = query(chat)#query({"inputs": chat, "parameters": llm_params, "options": {"use_cache": False}})
#         #print(response)
#         if response == None:
#             print('CAUGHT JSON ERROR')
#             continue

#         if type(response)==dict:
#             print("AN EXCEPTION")
#             time.sleep(2.5)
#             if "Inference Endpoints" in response['error']:
#               print("HOURLY RATE LIMIT REACHED")
#               time.sleep(900)
                
#         elif 'value' in response:
#             overloaded=0
    
#             response_split = response.split(";")
#             response_split = response_split[0].split(": ")
#             if len(response_split)<2:
#                 overloaded = 1
#     print(response_split[1])
#     return response_split[1]