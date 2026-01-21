import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import time
from cuda_check import CudaChecker

import config

# =========================
print("=" * 50)
print("ЗАПУСК ОБУЧЕНИЯ ruGPT-2")
print("=" * 50)

# =========================
# Проверка CUDA
# =========================

print("Проверка используемого при обучении устройства...")
cuda = CudaChecker()
cuda.check()

# =========================
# 1. Загрузка токенизатора
# =========================
print("\n[1/7] Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("Токенизатор загружен!")

# =========================
# 2. Загрузка модели
# =========================

print("\n[2/7] Загрузка модели...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
print("Модель загружена!")

# =========================
# 3. Загрузка датасета
# =========================

print("\n[3/7] Загрузка датасета...")
dataset = load_dataset(
    "text",
    data_files={"train": DATASET_PATH}
)
print(f"Загружено строк: {len(dataset['train'])}")

# =========================
# 4. Токенизация
# =========================
print("\n[4/7] Токенизация датасета...")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=config.MAX_LENGTH
    )

tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

print("Токенизация завершена")

# =========================
# 5. DataLocator
# =========================

print("\n[5/7] Подготовка DataCollator...")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
print("DataCollator готов")

# =========================
# 6. Настройки обучения
# =========================

print("\n[6/7] Настройка параметров обучения...")

training_args = TrainingArguments(
    output_dir=config.OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=config.EPOCHS,
    per_device_train_batch_size=config.BATCH_SIZE,
    learning_rate=config.LR,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
)

print("TrainingArguments готовы")

# =========================
# 7. Тренировка модели
# =========================

print("\n[7/7] Инициализация Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

print("\nНАЧИНАЕМ ОБУЧЕНИЕ\n")

start_time = time.time()
trainer.train()
end_time = time.time()

print("\nОБУЧЕНИЕ ЗАВЕРШЕНО")
print(f"Время обучения: {(end_time - start_time)/60:.2f} минут")

# =========================
# СОХРАНЕНИЕ
# =========================
print("\nСохранение модели...")
trainer.save_model(config.OUTPUT_DIR)
tokenizer.save_pretrained(config.OUTPUT_DIR)
print("Модель сохранена!")