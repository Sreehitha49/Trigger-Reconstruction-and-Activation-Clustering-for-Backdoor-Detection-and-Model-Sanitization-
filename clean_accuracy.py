import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import PreActResNet18

path = "data/cifar10_preactresnet18_badnet_0_01/attack_result.pt"
result = torch.load(path, map_location="cpu", weights_only=False)

model = PreActResNet18(num_classes=10)
model.load_state_dict(result["model"], strict=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

dataset = torchvision.datasets.CIFAR10(root="data", train=False,
                                       download=False, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print("Total clean test images:", total)
print("Correct predictions:", correct)
print(f"Clean Test Accuracy: {100 * correct / total:.2f}%")