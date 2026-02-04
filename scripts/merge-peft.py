from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig

# Define the base model and adapter IDs
base_model_id = "TildeAI/TildeOpen-30b"
adaptor_id = "tildeopen-30b-lora-adapter-enlvlt-2/checkpoint-3483"



# Reload the base model
base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,  # Use higher precision for consistency
    return_dict=True,
    low_cpu_mem_usage=True,
    device_map="auto",
    trust_remote_code=True,
)

# Load the adapter
peft_model = PeftModel.from_pretrained(base_model_reload, adaptor_id)

# Merge the adapter with the base model and unload the adapter
peft_model = peft_model.merge_and_unload()

# Reload the tokenizer
tokenizer_reload = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer_reload.pad_token = tokenizer_reload.eos_token
tokenizer_reload.padding_side = "right"


# Define the merged model ID
merge_model_id = "TildeOpen-30b-LatLit-instruct"

# Save the merged model and tokenizer
peft_model.save_pretrained(merge_model_id, push_to_hub=True)
tokenizer_reload.save_pretrained(merge_model_id, push_to_hub=True)



# Define the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Set to True for 8-bit quantization
    llm_int8_threshold=6.0
)

# Load the merged model with quantization
merged_model_quantized = AutoModelForCausalLM.from_pretrained(
    merge_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
)

# Save the quantized model
quantized_model_id = merge_model_id + "_quantized"
merged_model_quantized.save_pretrained(quantized_model_id, push_to_hub=True)