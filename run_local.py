#%%
import yaml
from munch import munchify
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
import huggingface_hub
huggingface_hub.login(config.model.API_TOKEN)
print('Start', flush=True)
import sys
import torch
print(f'torch available: {torch.cuda.is_available()}', flush=True)
print(torch.version.cuda)
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i))
from torch import cuda, bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import transformers
print(f'torch available: {torch.cuda.is_available()}', flush=True)
import bitsandbytes
import accelerate
print(f'python: {sys.version}', flush=True)
print(f'torch: {torch.__version__}', flush=True)
print(f'transformers: {transformers.__version__}', flush=True)
print(f'bitsandbytes: {bitsandbytes.__version__}', flush=True)
print(f'accelerate: {accelerate.__version__}', flush=True)
model_name = config.model.model_name
print(f'model: {model_name}')
import time
import numpy as np
import gc
import utils as ut
# from accelerate.utils import release_memory
# %%

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

flush()
quantized = config.model.quantized
# loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.model.API_TOKEN)

if not quantized:
    # full model
    print('Loading full model', flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, resume_download = True,token=config.model.API_TOKEN, cache_dir = '/mnt/shared_drive/llm_garage/cache/huggingface')#, local_files_only = True)
    model = model.to('cuda')
    model.config.use_cache = False
else:
    # quantized version
    print('Loading quantized model', flush=True)
    bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, resume_download = True, device_map='cuda:0', quantization_config=bnb_config, cache_dir = '/mnt/shared_drive/llm_garage/cache/huggingface')#, local_files_only = True)
    model.config.use_cache = False

def query(text, temperature = config.params.temperature, max_new_tokens = 6):
    if config.model.chat_template_is_avail:
        inputs = tokenizer.apply_chat_template(text, return_tensors="pt", continue_final_message=True).to("cuda:0")
    else:
        #inputs = tokenizer(text, return_tensors = "pt").to("cuda:0")
        inputs = tokenizer.encode(text, return_tensors = "pt").to("cuda:0")
    #print(inputs)
    with torch.no_grad():
        if temperature == 0:
            outputs = model.generate(inputs, max_new_tokens = max_new_tokens,output_scores=True, return_dict_in_generate=True, output_hidden_states=True, do_sample = False, pad_token_id=tokenizer.eos_token_id)
        else:
            outputs = model.generate(inputs, max_new_tokens = max_new_tokens,output_scores=True, return_dict_in_generate=True, output_hidden_states=True, do_sample = True, temperature = temperature, pad_token_id=tokenizer.eos_token_id)

    generated_tokens = outputs.sequences[0]
    prompt_length = inputs.shape[1]
    generated_text = tokenizer.decode(generated_tokens[prompt_length:], skip_special_tokens=True)
    #print(f'{generated_text}', flush = True)
    return {'generated_text': generated_text}#, 'generated_tokens': generated_tokens, 'outputs': outputs}

def get_response(chat, options):
    """Generate a response from the model."""

    overloaded = 1
    while overloaded == 1:
        response = query(text = chat)
        #print(response)
                
        if any(option in response['generated_text'].split("'") for option in options):
            overloaded=0
    response_split = response['generated_text'].split("'")
    for opt in options:
        try:
            index = response_split.index(opt)
        except:
            continue
    return response_split[index]

def get_meta_response(chat):
    """Generate a response from the model."""

    overloaded = 1
    #while overloaded == 1:
    response = query(text = chat)
      #if 'value' in response['generated_text']:
        #overloaded = 0
    print(response['generated_text'], flush=True)
    #print(response['generated_text'][len(chat)-5:], flush = True)
    return response['generated_text']

def encode_decode_options(options):
    target_encodings = [tokenizer.encode(option, add_special_tokens=False) for option in options]
    target_phrase = [[tokenizer.decode(target_token_id, skip_special_tokens=True) for target_token_id in encoding] for encoding in target_encodings]
    #target_phrase = [tokenizer.decode(target_id, skip_special_tokens=True) for target_id in target_ids]
    first_target_phrase = [target[0] for target in target_phrase]
    first_target_encoding = [target[0] for target in target_encodings]
    print(f"Options tokens: {target_phrase}")
    print(f"first options tokens: {first_target_phrase}")
    first_target_id_dict = {option: first_target_encoding[i] for i, option in enumerate(options)}
    return first_target_id_dict

def get_probability_dict(options, prompt, first_target_id_dict, temperature = config.params.temperature, epsilon=np.finfo(float).eps):
    
    # first, we need to generate the text!
    if config.model.chat_template_is_avail:
        inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True).to("cuda:0")
    else:
        inputs = tokenizer.encode(prompt, return_tensors = "pt").to("cuda:0")
        #inputs = tokenizer(prompt, return_tensors = "pt").to("cuda:0") #returns different inputs structure, , where model input is inputs['input_ids']

    with torch.no_grad():
        if temperature == 0:
            outputs = model.generate(inputs, max_new_tokens = 5,output_scores=True, return_dict_in_generate=True, output_hidden_states=True, do_sample = False, pad_token_id=tokenizer.eos_token_id)
        else:
            outputs = model.generate(inputs, max_new_tokens = 5,output_scores=True, return_dict_in_generate=True, output_hidden_states=True, do_sample = True, temperature = temperature, top_k = len(options), pad_token_id=tokenizer.eos_token_id)
    
    #generated_tokens = outputs.sequences[0]
    #generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    #print(generated_text)

    first_target_encoding = [first_target_id_dict[option] for option in options]
    probability_dict = {opt: -np.inf for opt in options}
    options_log_probs = []
    
    # find generation probability of first token in each action label. Make sure that these tokens are different for the possible action labels!
    for choice in first_target_encoding:

        logits = outputs.scores[0]
        probabilities = torch.log_softmax(logits, dim=-1)
        target_log_prob = probabilities[0, choice].item()
        # if np.exp(target_log_prob) < epsilon:
        #     target_log_prob = -np.inf
        target_log_prob = max(np.log(epsilon), target_log_prob)
        
        options_log_probs.append(target_log_prob)

    # normalize log prob over all choice probabilities for this configuration and prompt
    if -np.inf in options_log_probs:
        normed_probs = ut.normalize_probs(np.exp(options_log_probs))
        normed_log_probs = np.log(normed_probs)
    else:
        normed_log_probs = ut.normalize_logprobs(options_log_probs)
    
    for option, prob in zip(options, normed_log_probs):
        probability_dict[option] = prob
    return probability_dict


# LEGACY CODE (SAGAR YOU CAN IGNROE THIS)

#%%
# import yaml
# from munch import munchify
# with open("config.yaml", "r") as f:
#     doc = yaml.safe_load(f)
# config = munchify(doc)
# import huggingface_hub
# huggingface_hub.login(config.model.API_TOKEN)
# print('Start', flush=True)
# import sys
# import torch
# print(f'torch available: {torch.cuda.is_available()}', flush=True)
# print(torch.version.cuda)
# for i in range(torch.cuda.device_count()):
#    print(torch.cuda.get_device_properties(i))
# from torch import cuda, bfloat16
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import BitsAndBytesConfig
# import transformers
# print(f'torch available: {torch.cuda.is_available()}', flush=True)
# import bitsandbytes
# import accelerate
# print(f'python: {sys.version}', flush=True)
# print(f'torch: {torch.__version__}', flush=True)
# print(f'transformers: {transformers.__version__}', flush=True)
# print(f'bitsandbytes: {bitsandbytes.__version__}', flush=True)
# print(f'accelerate: {accelerate.__version__}', flush=True)
# model_name = config.model.model_name
# print(f'model: {model_name}')
# import time
# # %%
# quantized = config.model.quantized
# # loading tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.model.API_TOKEN)
# if not quantized:
#     # full model
#     print('Loading full model', flush=True)
#     model = AutoModelForCausalLM.from_pretrained(model_name, resume_download = True,token=config.model.API_TOKEN)#, local_files_only = True)
#     model = model.to('cuda')
#     model.config.use_cache = False
#     #pipeline = transformers.pipeline('text-generation', model=model, tokenizer=tokenizer, torch_dtype=torch.float16, device_map="auto")
# else:
#     # quantized version
#     print('Loading quantized model', flush=True)
#     bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)
#     model = AutoModelForCausalLM.from_pretrained(model_name, resume_download = True, device_map='cuda:0', quantization_config=bnb_config)#, local_files_only = True)
#     model.config.use_cache = False
#     #pipeline = transformers.pipeline('text-generation', model=model, tokenizer=tokenizer, device_map="auto")

# def query(text, temperature = config.params.temperature, max_new_tokens = 15):
#     inputs = tokenizer.apply_chat_template(text, return_tensors="pt", add_generation_prompt=True).to("cuda:0")
#     #print(inputs)
#     with torch.no_grad():
#         outputs = model.generate(inputs, max_new_tokens = max_new_tokens, temperature =temperature, return_dict_in_generate=True, output_hidden_states=True, do_sample = True)
#         generated_tokens = outputs.sequences[0]
#         prompt_length = inputs.shape[1]
#         generated_text = tokenizer.decode(generated_tokens[prompt_length:], skip_special_tokens=True)
        
#     return {'generated_text': generated_text}#, 'generated_tokens': generated_tokens, 'outputs': outputs}

# def get_response(chat, options):
#     """Generate a response from the model."""

#     overloaded = 1
#     while overloaded == 1:
#         response = query(text = chat)
#         #print(response)
                
#         if any(option in response['generated_text'].split("'") for option in options):
#             overloaded=0
#     response_split = response['generated_text'].split("'")
#     for opt in options:
#         try:
#             index = response_split.index(opt)
#         except:
#             continue
#     #print(response_split[index])
#     return response_split[index]

# def get_meta_response(chat):
#     """Generate a response from the model."""

#     overloaded = 1
#     while overloaded == 1:
#         response = query(text = chat)
#         #print(response)
#         if 'value' in response['generated_text']:
#             overloaded=0
    
#             response_split = response['generated_text'].split(";")
#             response_split = response_split[0].split(": ")
#             if len(response_split)<2:
#                 overloaded = 1
#     print(response_split[1])
#     #time.sleep(5)
#     return response_split[1]
