import torch # type: ignore
import torch.nn as nn # type: ignore

class LoRAAdapter(nn.Module):
    def __init__(self, base_model, rank=4):
        super(LoRAAdapter, self).__init__()
        self.base_model = base_model
        self.rank = rank
        self.adapters = nn.ModuleDict()
        
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):  # Apply LoRA to Linear layers
                self.adapters[name] = nn.Linear(module.in_features, rank, bias=False)
                self.adapters[name + "_output"] = nn.Linear(rank, module.out_features, bias=False)
        
    def forward(self, x):
        for name, module in self.base_model.named_modules():
            if name in self.adapters:
                x = module(x) + self.adapters[name](x) @ self.adapters[name + "_output"].weight
            else:
                x = module(x)
        return x
