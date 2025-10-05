from dataclasses import dataclass
from pathlib import Path


# Default paths (adjust as needed)
BT_REWARD ="danielfein/bt-rewmodel-meta-llama_Llama-3.1-8B-final-20250512_094156"
EMBEDING_MODEL = "/pscratch/sd/r/ritesh11/temp/Qwen3-Embedding-0.6B"
MODEL_PATH = Path("/pscratch/sd/r/ritesh11/temp/models/litbench_reward")


BASE_DIR = Path("./")
FICTIONAL_PROMPTS_YAML = BASE_DIR / "fictional_prompts.yaml"
