from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass
class TrainingConfig:
    dataset: Path = PACKAGE_DIR / "d.txt"
    output_dir: Path = PACKAGE_DIR / "trained_models"
    model: str | Path = (
        "ai-forever/rugpt3medium_based_on_gpt2"
    )
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 4e-5
    max_length: int = 256
