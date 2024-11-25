import torch # type: ignore

def validate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()
    
    print(f"Validation Loss: {total_loss / len(dataloader)}")
