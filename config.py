"""Configuration file for path variables."""
from pathlib import Path

# Base directory
BASE_DIR = Path("./")

# Input paths
PROMPTS_YAML = BASE_DIR / "non_fictional_prompts.yaml"

# Output paths
OUTPUT_DIR_BASE_PATH = Path("/mnt/d/code/open_source/mats/thought_anchors_writing/data")
OUTPUT_DIR = OUTPUT_DIR_BASE_PATH / "generated_samples"

# Model configuration
MODEL_NAME = "qwen/qwen3-14b:free" 
