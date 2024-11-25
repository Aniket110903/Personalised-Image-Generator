import torch # type: ignore

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model, load_path, device):
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()
    return model
