
import torch

path = "data/cifar10_preactresnet18_badnet_0_01/attack_result.pt"

result = torch.load(
    path,
    map_location="cpu",
    weights_only=False
)

print("Model name:", result["model_name"])
print("Number of classes:", result["num_classes"])

model_weights = result["model"]

print("Total parameter tensors:", len(model_weights))
print("\nFirst 10 parameter names:")

for i, (name, value) in enumerate(model_weights.items()):
    if i == 10:
        break
    print(name, tuple(value.shape))

print("\nModel weights loaded successfully!")