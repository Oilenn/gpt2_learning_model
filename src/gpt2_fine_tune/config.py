from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    dataset: Path = Path("dataset")
    output_dir: Path = Path("trained_models")
    model: str = "sberbank-ai/rugpt3medium_based_on_gpt2"
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 4e-5
    max_length: int = 256