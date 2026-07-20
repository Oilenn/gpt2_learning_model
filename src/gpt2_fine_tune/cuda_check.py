import torch

def get_device():
    """Определяет лучшее доступное устройство"""
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        print(f"Используемое устройство CUDA: {device_name}")
        return torch.device('cuda')
    else:
        print("Используется CPU...")
        return torch.device('cpu')

def check():
    """Основная проверка (для совместимости)"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    return torch.cuda.is_available()
