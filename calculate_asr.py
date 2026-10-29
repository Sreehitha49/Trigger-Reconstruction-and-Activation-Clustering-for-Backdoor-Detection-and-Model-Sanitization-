import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from model import PreActResNet18

DATA_DIR = "data/cifar10_preactresnet18_badnet_0_01/bd_test_dataset"
MODEL_PATH = "data/cifar10_preactresnet18_badnet_0_01/attack_result.pt"
TARGET_CLASS = 0

class PoisonedDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
        for folder in sorted(os.listdir(root)):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue
            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(folder_path, filename), int(folder)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, original_label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), original_label

result = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model = PreActResNet18(num_classes=10)
model.load_state_dict(result["model"], strict=True)
model.eval()

loader = DataLoader(PoisonedDataset(DATA_DIR), batch_size=128, shuffle=False)

total = 0
hits = 0

with torch.no_grad():
    for images, original_labels in loader:
        predictions = model(images).argmax(dim=1)
        eligible = original_labels != TARGET_CLASS
        hits += ((predictions == TARGET_CLASS) & eligible).sum().item()
        total += eligible.sum().item()

print("Images evaluated:", total)
print("Predicted as target class:", hits)
print(f"Trigger Success Rate (ASR): {100 * hits / total:.2f}%")