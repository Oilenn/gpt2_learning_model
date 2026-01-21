import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.cuda_data import Cuda_data

import config

class LLM:
    tokenizer = None
    model = None

    def __init__(self):
        self.cuda = Cuda_data()
        self.cuda.check()
        self.load_model()

    def load_model(self):
        print("Загружаем модель...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(config.MODEL_PATH)

        # Исправлено: используем self.cuda.device как объект, а не функцию
        self.model.to(self.cuda.device)
        self.model.eval()
        print("Модель загружена!")

    # =========================
    # ГЕНЕРАЦИЯ
    # =========================
    def generate_answer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.cuda.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Берем только ответ после промпта
        response = text[len(prompt):]

        # Удаляем "user:" и "assistant:" (любой регистр)
        import re
        response = re.sub(r'(?i)(user:|assistant:)', '', response)

        # Убираем лишние пробелы в начале
        response = response.lstrip()

        # Удаляем запятую, тире или двоеточие в начале строки
        if response and response[0] in ',:-—':
            response = response[1:].lstrip()

        # Убираем лишние пробелы и переносы
        response = response.strip()

        # Убираем множественные переносы
        while '\n\n' in response:
            response = response.replace('\n\n', '\n')

        # Если ответ пустой
        if not response:
            return "🤔"

        return response
