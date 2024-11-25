import torch # type: ignore
from flux import FluxModel  # type: ignore  Replace with the correct Flux model import

def load_base_model(pretrained=True):
    model = FluxModel(pretrained=pretrained)
    return model
