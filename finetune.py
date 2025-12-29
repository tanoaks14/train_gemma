import os
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login, HfApi


# ==========================================
# 1. Authentication & Configuration
# ==========================================
# IMPORTANT: You must accept the terms at https://huggingface.co/google/functiongemma-270m-it
# Replace 'your_hf_token_here' with your actual token or run `huggingface-cli login` in your terminal.
HF_TOKEN = os.getenv("HF_TOKEN", "PASTE_YOUR_TOKEN_HERE")

def setup_auth(token):
    if token == "PASTE_YOUR_TOKEN_HERE":
        print("❌ ERROR: You must provide a valid Hugging Face token in the script.")
        return None
    
    try:
        login(token=token)
        api = HfApi()
        user_info = api.whoami(token=token)
        print(f"✅ Authenticated as: {user_info['name']}")
        return token
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return None

active_token = setup_auth(HF_TOKEN)

model_id = "google/functiongemma-270m-it"
output_dir = "./functiongemma-finetuned"

# Local finetune inputs (externalized)
TOOLS_PATH = os.getenv("TOOLS_PATH", "./data/tools.json")
DATASET_PATH = os.getenv("DATASET_PATH", "./data/finetune_dataset.jsonl")

# 4-bit quantization (Ideal for 4060 Ti 16GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"🔄 Loading tokenizer and model: {model_id}...")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    token=active_token,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=active_token
)

# ==========================================
# 2. Tool Definition (JSON Schema)
# ==========================================
def load_tools(path: str):
    with open(path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    if not isinstance(tools, list):
        raise ValueError(f"Tools file must be a JSON list. Got: {type(tools)}")
    return tools

# ==========================================
# 3. PEFT (LoRA) Configuration
# ==========================================
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# ==========================================
# 4. Training Arguments (SFTConfig for TRL 0.12+)
# ==========================================
sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,   # Optimized for 4060 Ti 16GB
    gradient_accumulation_steps=8, 
    learning_rate=2e-4,
    logging_steps=5,
    max_steps=100,
    save_steps=50,
    bf16=True, 
    optim="paged_adamw_32bit",
    report_to="none",
    dataset_kwargs={"max_seq_length": 1024}
)

# ==========================================
# 5. Dataset Processing
# ==========================================
def formatting_prompts_func(example):
    """
    Revised formatting function for TRL's SFTTrainer.
    The function should return the formatted string directly for each row.
    """
    system_message = {
        "role": "developer",
        "content": "You are a model that can do function calling with the following functions"
    }

    messages = example["messages"]
    if messages and messages[0].get("role") in {"system", "developer"}:
        # If dataset already includes a system/developer message, keep it and don't prepend.
        combined_messages = messages
    else:
        combined_messages = [system_message] + messages

    tool_names = example.get("tool_names")
    tools_for_example = ALL_TOOLS
    if tool_names:
        name_set = set(tool_names)
        tools_for_example = [t for t in ALL_TOOLS if t.get("function", {}).get("name") in name_set]

    return tokenizer.apply_chat_template(
        combined_messages,
        tools=tools_for_example,
        tokenize=False,
        add_generation_prompt=False,
    )


print(f"🔄 Loading tools: {TOOLS_PATH}...")
ALL_TOOLS = load_tools(TOOLS_PATH)

print(f"🔄 Loading local dataset: {DATASET_PATH}...")
dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

# SFTTrainer handles the mapping internally when formatting_func is provided
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_config,                
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
)

if __name__ == "__main__":
    if active_token:
        print("🚀 Starting Fine-tuning on RTX 4060 Ti...")
        trainer.train()
        trainer.save_model(output_dir)
        print(f"✅ Model saved to {output_dir}")
    else:
        print("🛑 Script stopped due to authentication error.")