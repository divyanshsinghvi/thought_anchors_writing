"""Configuration file for twist ablation study."""
from pathlib import Path

# Base directory for twist study
TWIST_BASE_DIR = Path(__file__).parent

# Input paths
TWIST_PROMPTS_YAML = TWIST_BASE_DIR / "prompts.yaml"

# Output paths
OUTPUT_DIR_BASE_PATH = Path("/mnt/d/code/open_source/mats/thought_anchors_writing/data/twist")
OUTPUT_DIR_TWIST = OUTPUT_DIR_BASE_PATH / "stories"

# Cache path
CACHE_PATH = OUTPUT_DIR_BASE_PATH / "cache"

# Model configuration
MODEL_NAME = "qwen/qwen3-14b"

# Generation configuration
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_N_SAMPLES = 1

# Story parameters
TARGET_STORY_LENGTH = "3-5 sentences"
