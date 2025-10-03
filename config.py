"""Configuration file for path variables."""
from pathlib import Path

# Base directory
BASE_DIR = Path("./")

# Input paths
PROMPTS_YAML = BASE_DIR / "non_fictional_prompts.yaml"
FICTIONAL_PROMPTS_YAML = BASE_DIR / "fictional_prompts.yaml"

# Output paths
OUTPUT_DIR_BASE_PATH = Path("/mnt/d/code/open_source/mats/thought_anchors_writing/data")
OUTPUT_DIR_NON_FICTION = OUTPUT_DIR_BASE_PATH / "non_fiction"
OUTPUT_DIR_FICTION = OUTPUT_DIR_BASE_PATH / "fiction"

# Cache path
CACHE_PATH = OUTPUT_DIR_BASE_PATH / "cache"

# Model configuration
# MODEL_NAME = "qwen/qwen3-14b:free"
MODEL_NAME = "qwen/qwen3-14b"

# Evaluation configuration
DEFAULT_EVALUATOR_MODELS = [
    "qwen/qwen3-14b:free",
    "moonshotai/kimi-k2-0905",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "deepseek/deepseek-r1-0528:free",
    # Add more evaluator models here to reduce bias
    # "Qwen/Qwen2.5-72B-Instruct",
    # "meta-llama/llama-3.1-70b-instruct",
]
DEFAULT_EVALUATION_TEMPERATURE = 0.3
DEFAULT_EVALUATION_SAMPLES = 1
EVALUATED_SUFFIX = "_evaluated" 
