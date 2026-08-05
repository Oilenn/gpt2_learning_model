import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import TrainingConfig

class LLM:
    tokenizer = None
    model = None

    def __init__(self):
        self.load_model()

    def load_model(self):
        print("Загружаем модель...")
        self.tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.model)
        self.model = AutoModelForCausalLM.from_pretrained(TrainingConfig.model)

        # Исправлено: используем self.cuda.device как объект, а не функцию
        self.model.eval()
        print("Модель загружена!")

    # =========================
    # ГЕНЕРАЦИЯ
    # =========================
    def generate_answer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,
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

if __name__ == "__main__":
    l = LLM()
    print(l.generate_answer())