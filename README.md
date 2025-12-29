# FunctionGemma-270M MCP Tools Finetuning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/google/functiongemma-270m-it)

Fine-tune Google's FunctionGemma-270M model on custom Model Context Protocol (MCP) tools for advanced function calling capabilities. This project provides a complete pipeline to scan MCP tools, create training datasets, and fine-tune the model using efficient 4-bit quantization and LoRA (Low-Rank Adaptation).

## ✨ Features

- 🔍 **Automatic MCP Tool Scanner** - Recursively scans Python codebases to extract tool definitions and convert them to OpenAI-compatible function schemas
- 🎯 **Efficient Fine-tuning** - Uses 4-bit quantization (QLoRA) for training on consumer GPUs (16GB VRAM)
- 📊 **Dataset Generation** - Converts tool definitions into properly formatted training examples
- 🚀 **Optimized for RTX 4060 Ti** - Configured for 16GB VRAM with gradient accumulation
- 🛠️ **Full Pipeline** - From tool scanning to model deployment

## 🎯 Use Cases

- Train language models to understand and call your custom tools
- Adapt FunctionGemma to domain-specific APIs and functions
- Create specialized AI assistants with custom function calling capabilities
- Build MCP-compatible tool-using agents

## 📋 Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with 16GB+ VRAM (recommended: RTX 4060 Ti)
- [Hugging Face account](https://huggingface.co/) with access to [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
- Hugging Face token with read access

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/functiongemma-mcp-finetune.git
cd functiongemma-mcp-finetune
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv gemma_finetune_env

# Activate (Windows)
gemma_finetune_env\Scripts\activate

# Activate (Linux/Mac)
source gemma_finetune_env/bin/activate

# Install dependencies
pip install torch transformers datasets peft trl bitsandbytes accelerate huggingface_hub
```

### 3. Configure Hugging Face Authentication

```bash
# Option 1: Use environment variable
set HF_TOKEN=your_hf_token_here

# Option 2: Edit finetune.py and replace PASTE_YOUR_TOKEN_HERE
```

**Important:** Accept the terms at https://huggingface.co/google/functiongemma-270m-it before running.

### 4. Scan MCP Tools (Optional)

If you have your own MCP tools codebase to scan:

```bash
python scan_mcp_tools.py /path/to/your/codebase --output data/tools.json
```

This will recursively scan Python files and extract tool definitions.

### 5. Prepare Training Dataset

Place your training examples in `data/finetune_dataset.jsonl` in the format:

```json
{
  "id": "example-1",
  "tool_names": ["get_weather"],
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}]},
    {"role": "tool", "name": "get_weather", "content": "{\"temperature\": 18, \"conditions\": \"cloudy\"}"},
    {"role": "assistant", "content": "It's 18°C and cloudy in Paris."}
  ]
}
```

### 6. Run Fine-tuning

```bash
python finetune.py
```

The script will:
- Load the base FunctionGemma-270M model
- Apply 4-bit quantization for memory efficiency
- Fine-tune using LoRA adapters
- Save checkpoints to `functiongemma-finetuned/`

## 📁 Project Structure

```
.
├── finetune.py                 # Main fine-tuning script
├── scan_mcp_tools.py           # Tool definition scanner
├── data/
│   ├── tools.json              # Extracted tool definitions
│   └── finetune_dataset.jsonl  # Training examples
├── functiongemma-finetuned/    # Output directory (git-ignored)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── checkpoint-*/
└── tests/
    └── test_scan_mcp_tools.py  # Unit tests
```

## ⚙️ Configuration

### Training Parameters

Edit these in `finetune.py`:

```python
sft_config = SFTConfig(
    per_device_train_batch_size=2,    # Batch size per GPU
    gradient_accumulation_steps=8,    # Effective batch size = 2 * 8 = 16
    learning_rate=2e-4,               # Learning rate
    max_steps=100,                    # Total training steps
    save_steps=50,                    # Checkpoint frequency
    bf16=True,                        # Use bfloat16 precision
)
```

### LoRA Configuration

```python
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,                             # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", ...],
)
```

## 🔍 Tool Scanner Details

The `scan_mcp_tools.py` script automatically extracts:
- Function names and descriptions (from docstrings)
- Parameter types and descriptions
- Required vs optional parameters
- Nested type annotations (List, Dict, Optional, etc.)

**Example:**

```python
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get current weather for a city.
    
    Args:
        city: The city name
        unit: Temperature unit (celsius or fahrenheit)
    """
    pass
```

Converts to:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "The city name"},
        "unit": {"type": "string", "description": "Temperature unit"}
      },
      "required": ["city"]
    }
  }
}
```

## 🎓 Training Tips

1. **GPU Memory Issues?**
   - Reduce `per_device_train_batch_size` to 1
   - Increase `gradient_accumulation_steps` to 16
   - Reduce `max_seq_length` in dataset_kwargs

2. **Better Results?**
   - Increase `max_steps` (100 → 500+)
   - Add more diverse training examples
   - Tune `learning_rate` (try 1e-4 or 5e-5)

3. **Faster Training?**
   - Use `fp16=True` instead of `bf16` (if supported)
   - Reduce `save_steps` to save less frequently

## 📊 Performance

**Hardware:** RTX 4060 Ti (16GB VRAM)
- **Training Speed:** ~5 steps/minute
- **Memory Usage:** ~14GB VRAM
- **Total Time:** ~20 minutes for 100 steps

## 🧪 Testing

Run unit tests for the tool scanner:

```bash
python -m pytest tests/test_scan_mcp_tools.py -v
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Google DeepMind](https://huggingface.co/google) for FunctionGemma-270M
- [Hugging Face](https://huggingface.co/) for transformers and PEFT libraries
- [TRL](https://github.com/huggingface/trl) for supervised fine-tuning utilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or issues, please open a GitHub issue or reach out via discussions.

## ⭐ Star History

If this project helps you, please consider giving it a star! ⭐

---

**Made with ❤️ for the AI community**
