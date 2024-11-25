import torch # type: ignore
from torch.optim import Adam # type: ignore
from tqdm import tqdm # type: ignore

def train_model(model, dataloader, epochs, lr, device):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()  # Replace with a relevant loss function

    for epoch in range(epochs):
        epoch_loss = 0
        for images, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)  # Autoencoder-style
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)}")
