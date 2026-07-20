import cuda_check
import config

cuda_check.get_device()

print("=" * 50)
print("ЗАПУСК ТОНКОЙ НАСТРОЙКИ НЕЙРОСЕТИ")
print("=" * 50)

print("Проверка используемого при обучении устройства...")
cuda_check.get_device()

