"""Configuration file for path variables."""
from pathlib import Path

# Base directory
BASE_DIR = Path("./")

# Input paths
PROMPTS_YAML = BASE_DIR / "non_fictional_prompts.yaml"

# Output paths
OUTPUT_DIR_BASE_PATH = Path("/mnt/d/code/open_source/mats/thought_anchors_writing/data")
OUTPUT_DIR_NON_FICTION = OUTPUT_DIR_BASE_PATH / "non_fiction"
OUTPUT_DIR_FICTION = OUTPUT_DIR_BASE_PATH / "fiction"

# Cache path
CACHE_PATH = OUTPUT_DIR_BASE_PATH / "cache"

# Model configuration
MODEL_NAME = "qwen/qwen3-14b:free" 
