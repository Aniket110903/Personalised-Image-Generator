import torch # type: ignore
from src.models.base_model import load_base_model
from src.models.lora_adapter import LoRAAdapter
from src.datasets.data_loader import get_dataloader
from src.training.train import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    base_model = load_base_model().to(device)
    lora_model = LoRAAdapter(base_model).to(device)
    dataloader = get_dataloader(data_dir='data/train', batch_size=16, img_size=256)
    train_model(lora_model, dataloader, epochs=10, lr=1e-4, device=device)
