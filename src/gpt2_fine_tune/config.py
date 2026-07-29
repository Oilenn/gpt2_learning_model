from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass
class TrainingConfig:
    dataset: Path = PACKAGE_DIR / "s.txt"
    output_dir: Path = PACKAGE_DIR / "trained_models"
    model: str | Path = PACKAGE_DIR / "checkpoint"
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 4e-5
    max_length: int = 256
