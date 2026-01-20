import torch


class CudaChecker:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device = self._get_device()

    def _get_device(self):
        """Определяет лучшее доступное устройство"""
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_id)
            print(f"Используемое устройство CUDA: {device_name}")
            return torch.device('cuda')
        else:
            print("Используется CPU...")
            return torch.device('cpu')

    def check(self):
        """Основная проверка (для совместимости)"""
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {self.cuda_available}")

        if self.cuda_available:
            print(f"Device: {torch.cuda.get_device_name(0)}")

        return self.device
