from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Paths to model artifacts used by the evaluation pipeline."""

    bt_reward: Path
    tiger_score: Path


# Default paths (adjust as needed)
MODEL_PATHS = ModelConfig(
    bt_reward="danielfein/bt-rewmodel-meta-llama_Llama-3.1-8B-final-20250512_094156",
    tiger_score=Path("/pscratch/sd/r/ritesh11/temp/models/TIGERScore-13B"),
)


