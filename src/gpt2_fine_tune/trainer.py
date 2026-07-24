from pathlib import Path

from datasets import load_dataset

import cuda_check
from config import TrainingConfig

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

class FineTuner:
    """Класс для начала тонкой настройки"""
    def __init__(self, dataset: Path | str,config: TrainingConfig | None):
        self.dataset = dataset
        self.config = config
        self.epochs: int = 1
        self.batch_size: int = 8
        self.learning_rate: float = 4e-5

    def start_train(self):
        """Запуск тонкой настройки"""
        cuda_check.get_device()

        print("=" * 50)
        print("ЗАПУСК ТОНКОЙ НАСТРОЙКИ НЕЙРОСЕТИ")
        print("=" * 50)

    def with_epochs(self, epochs: int):
        """Задание эпох"""
        self.epochs = epochs
        return self

    def with_batch_size(self, batch_size: int):
        """Задание размера батча"""
        self.batch_size = batch_size
        return self

    def with_learning_rate(self, learning_rate: float):
        """Задание размера обучения"""
        self.learning_rate = learning_rate
        return self

    def with_dataset(self, dataset: Path | str):
        """Задание размера обучения"""
        if type(dataset) == str:
            dataset = Path(dataset)
        self.dataset = dataset
        return self

    def with_config(self, config: TrainingConfig | None):
        self.config = config
        return self

class TrainTune:
    def __init__(self, fineTuner: FineTuner):
        self._finetuner = fineTuner
        self.tokenizer = None
        self.model = None

    def train(self):
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    def load_tokenizer(self):
        print("\n[1/7] Загрузка токенизатора...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._finetuner.config.model)
            tokenizer.pad_token = tokenizer.eos_token
        except:
            raise ValueError("Ошибка при загрузке токенизатора!")
        print("Токенизатор загружен!")
        return tokenizer

    def load_model(self):
        print("\n[2/7] Загрузка модели...")
        try:
            model = AutoModelForCausalLM.from_pretrained(self._finetuner.config.model)
            model.resize_token_embeddings(len(self.tokenizer))
        except:
            raise ValueError("Ошибка загрузки модели!")
        print("Модель загружена!")
        return model

    def load_dataset(self):
        print("\n[3/7] Загрузка датасета...")
        try:
            dataset = load_dataset(
                "text",
                data_files={"train": self._finetuner.config.dataset}
            )
        except:
            raise ValueError("Ошибка загрузки датасета!")
        print(f"Датасет загружен!")
        print(f"Загружено строк: {len(dataset['train'])}")


"""

cuda_check.get_device()

print("=" * 50)
print("ЗАПУСК ТОНКОЙ НАСТРОЙКИ НЕЙРОСЕТИ")
print("=" * 50)

print("Проверка используемого при обучении устройства...")
cuda_check.get_device()

print("\n[1/7] Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(config.model)
tokenizer.pad_token = tokenizer.eos_token
print("Токенизатор загружен!")
"""