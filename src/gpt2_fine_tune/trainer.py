import time
from pathlib import Path

import cuda_check
from config import TrainingConfig

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


class FineTuner:
    """Класс для начала тонкой настройки"""

    def __init__(
        self,
        dataset: Path | str,
        config: TrainingConfig | None = None,
    ):
        self.config = config or TrainingConfig()
        self.config.dataset = Path(dataset).resolve()

    def start_train(self):
        """Запуск тонкой настройки"""
        cuda_check.get_device()

        print("=" * 50)
        print("ЗАПУСК ТОНКОЙ НАСТРОЙКИ НЕЙРОСЕТИ")
        print("=" * 50)

        train_tune = TrainTune(self.config)
        return train_tune.start_train()

    def with_epochs(self, epochs: int):
        """Задание эпох"""
        if epochs <= 0:
            raise ValueError("Количество эпох должно быть больше нуля")
        self.config.epochs = epochs
        return self

    def with_batch_size(self, batch_size: int):
        """Задание размера батча"""
        if batch_size <= 0:
            raise ValueError("Размер батча должен быть больше нуля")
        self.config.batch_size = batch_size
        return self

    def with_learning_rate(self, learning_rate: float):
        """Задание скорости обучения"""
        if learning_rate <= 0:
            raise ValueError("Скорость обучения должна быть больше нуля")
        self.config.learning_rate = learning_rate
        return self

    def with_dataset(self, dataset: Path | str):
        """Задание пути к датасету"""
        self.config.dataset = Path(dataset).resolve()
        return self

    def with_config(self, config: TrainingConfig):
        """Изменение конфига"""
        self.config = config
        return self


class TrainTune:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.data_collator = None
        self.trainer = None

    def start_train(self):
        self.validate_config()
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()
        self.dataset = self.load_dataset()
        self.tokenize_dataset()
        self.prepare_data_locator()
        self.set_training_args()
        self.train()
        self.save_model()
        return self.trainer

    def validate_config(self):
        if isinstance(self.config.model, Path) and not self.config.model.exists():
            raise FileNotFoundError(
                f"Каталог модели не найден: {self.config.model.resolve()}"
            )

        if not self.config.dataset.exists():
            raise FileNotFoundError(
                f"Датасет не найден: {self.config.dataset.resolve()}"
            )

    def load_tokenizer(self):
        print("\n[1/7] Загрузка токенизатора...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as error:
            raise RuntimeError(
                f"Не удалось загрузить токенизатор из {self.config.model!s}"
            ) from error

        print("Токенизатор загружен!")
        return self.tokenizer

    def load_model(self):
        print("\n[2/7] Загрузка модели...")
        try:
            model = AutoModelForCausalLM.from_pretrained(self.config.model)
            model.resize_token_embeddings(len(self.tokenizer))
        except Exception as error:
            raise RuntimeError(
                f"Не удалось загрузить модель из {self.config.model!s}"
            ) from error

        print("Модель загружена!")
        return model

    def load_dataset(self):
        print("\n[3/7] Загрузка датасета...")
        try:
            from datasets import load_dataset

            self.dataset = load_dataset(
                "text",
                data_files={"train": str(self.config.dataset)},
            )
        except Exception as error:
            raise RuntimeError(
                f"Не удалось загрузить датасет {self.config.dataset}"
            ) from error

        print("Датасет загружен!")
        print(f"Загружено строк: {len(self.dataset['train'])}")
        return self.dataset

    def tokenize_dataset(self):
        print("\n[4/7] Токенизация...")

        self.dataset = self.dataset.map(
            lambda batch: self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length
            ),
            batched=True,
            remove_columns=["text"]
        )

        print("Токенизация завершена!")

    def prepare_data_locator(self):
        print("\n[5/7] Подготовка DataCollator...")

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        print("DataCollator готов")

    def set_training_args(self):
        print("\n[6/7] Настройка параметров обучения...")

        self.training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            #logging_steps=100,
            #save_steps=500,
            #save_total_limit=2,
            #report_to="none",
            #logging_dir="./logs",
            fp16=torch.cuda.is_available()
        )

    def train(self):
        print("\n[7/7] Инициализация Trainer...")
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            data_collator=self.data_collator,
        )

        print("\nНАЧИНАЕМ ОБУЧЕНИЕ\n")

        start_time = time.time()
        self.trainer.train()
        end_time = time.time()

        print("\nОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"Время обучения: {(end_time - start_time) / 60:.2f} минут")


    def save_model(self):
        print("\nСохранение модели...")
        self.trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print("Модель сохранена!")


if __name__ == "__main__":
    fine_tuner = FineTuner(TrainingConfig().dataset)
    fine_tuner.start_train()
