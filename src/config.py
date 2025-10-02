from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Paths to model artifacts used by the evaluation pipeline."""

    bt_reward: Path
    tiger_score: Path


# Default paths (adjust as needed)
MODEL_PATHS = ModelConfig(
    bt_reward=Path("/pscratch/sd/r/ritesh11/temp/models/litbench_reward"),
    tiger_score=Path("/pscratch/sd/r/ritesh11/temp/models/TIGERScore-13B"),
)


