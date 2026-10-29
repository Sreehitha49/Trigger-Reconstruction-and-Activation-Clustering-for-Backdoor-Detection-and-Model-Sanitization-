import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# Step 1: Preprocess the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Step 2: Load CIFAR-10
dataset = datasets.CIFAR10(root="data", train=False,
                           download=False, transform=transform)

print("Dataset loaded successfully!")
print("Total images:", len(dataset))

# Step 3: Display one image (undo normalization for display only)
image, label = dataset[0]
mean = torch.tensor(MEAN).view(3, 1, 1)
std = torch.tensor(STD).view(3, 1, 1)
image = image * std + mean

plt.imshow(image.permute(1, 2, 0))
plt.title("Original Image")
plt.axis("off")
plt.show()