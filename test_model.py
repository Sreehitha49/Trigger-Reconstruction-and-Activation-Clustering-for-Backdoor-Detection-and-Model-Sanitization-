
import torch
from model import PreActResNet18

path = "data/cifar10_preactresnet18_badnet_0_01/attack_result.pt"

result = torch.load(
    path,
    map_location="cpu",
    weights_only=False
)

model = PreActResNet18(num_classes=10)

model.load_state_dict(result["model"], strict=True)
model.eval()

print("Model architecture matches the saved weights!")

dummy_input = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)
print("Predicted class:", output.argmax(dim=1).item())