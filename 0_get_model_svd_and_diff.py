import torch
from transformers import AutoModelForCausalLM

base_model = "meta-llama/Llama-2-13b-hf"
sft_model_list = [
    "WizardLMTeam/WizardLM-13B-V1.2"
]

