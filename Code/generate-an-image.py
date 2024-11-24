# Flux LoRA Training Implementation
# ----------------------------------
# This script includes environment setup, dataset preparation, model integration with LoRA,
# training execution, evaluation, and saving the trained model.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import json

# Environment Setup
# ------------------
# Ensure CUDA is available and set up the environment
print("CUDA Available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Preparation
# --------------------
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None):
        self.image_dir = image_dir
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_name = list(self.captions.keys())[idx]
        image_path = f"{self.image_dir}/{image_name}"
        image = Image.open(image_path).convert("RGB")
        caption = self.captions[image_name]

        if self.transform:
            image = self.transform(image)

        return image, caption

# Apply transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load Dataset
image_dir = "images"
captions_file = "captions.json"
dataset = CustomImageDataset(image_dir, captions_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# LoRA Integration
# -----------------
class LoRALayer(nn.Module):
    def __init__(self, input_dim, rank):
        super(LoRALayer, self).__init__()
        self.down_proj = nn.Linear(input_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, input_dim, bias=False)

    def forward(self, x):
        return x + self.up_proj(self.down_proj(x))

def apply_lora_to_flux(model, rank=4):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'attention' in name:
            lora_layer = LoRALayer(module.in_features, rank)
            module = nn.Sequential(module, lora_layer)
    return model

# Load Pre-trained Model and Apply LoRA
model = AutoModel.from_pretrained("black-forest-ai-labs/flux")
model = apply_lora_to_flux(model).to(device)

# Training Loop
# -------------
learning_rate = 1e-4
epochs = 10
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, captions in dataloader:
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, captions)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Evaluation
# ----------
model.eval()
val_loss = 0
with torch.no_grad():
    for images, captions in dataloader:
        images, captions = images.to(device), captions.to(device)
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, captions)
        val_loss += loss.item()

avg_val_loss = val_loss / len(dataloader)
print(f"Validation Loss: {avg_val_loss:.4f}")

# Save Trained Model
# ------------------
model.save_pretrained("fine-tuned-flux-lora")
print("Model saved successfully!")

# Test Model Output
# -----------------
tokenizer = AutoTokenizer.from_pretrained("black-forest-ai-labs/flux")
sample_caption = "A serene landscape with mountains and lakes in the style of impressionist painting"
inputs = tokenizer(sample_caption, return_tensors="pt").to(device)

with torch.no_grad():
    generated_image = model.generate(inputs["input_ids"])
    generated_image.show()