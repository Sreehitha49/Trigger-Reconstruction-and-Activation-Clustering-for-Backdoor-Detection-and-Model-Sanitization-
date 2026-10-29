
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
import torchvision.transforms as transforms
from model import PreActResNet18

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
SCALE = 8   # display only: 32x32 -> 256x256

path = "data/cifar10_preactresnet18_badnet_0_01/attack_result.pt"
result = torch.load(path, map_location="cpu", weights_only=False)

model = PreActResNet18(num_classes=10)
model.load_state_dict(result["model"], strict=True)
model.eval()

dataset = torchvision.datasets.CIFAR10(root="data", train=False,
                                       download=False,
                                       transform=transforms.ToTensor())

mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
std = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)

def predict(img):
    with torch.no_grad():
        return model(((img - mean) / std).unsqueeze(0)).argmax(1).item()

def upscale(img):
    up = F.interpolate(img.unsqueeze(0), scale_factor=SCALE,
                       mode="bicubic", align_corners=False)
    return up.squeeze(0).clamp(0, 1)

# Pick example images where the backdoor works (for illustration only)
examples = []
for idx in range(len(dataset)):
    image, label = dataset[idx]
    if label == 0:
        continue
    triggered = image.clone()
    triggered[:, 29:32, 29:32] = 1.0          # 3x3 white patch, bottom-right
    clean_pred, trig_pred = predict(image), predict(triggered)
    if clean_pred == label and trig_pred == 0:
        examples.append((image, label, clean_pred, trig_pred))
    if len(examples) == 3:
        break

fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), dpi=120)

for col, (image, label, cp, tp) in enumerate(examples):
    clean_up = upscale(image)
    trig_up = clean_up.clone()
    trig_up[:, 29 * SCALE:32 * SCALE, 29 * SCALE:32 * SCALE] = 1.0  # crisp patch

    axes[0, col].imshow(clean_up.permute(1, 2, 0).numpy())
    axes[0, col].set_title(f"Clean image\nPredicted: {CLASSES[cp]}", fontsize=12)

    axes[1, col].imshow(trig_up.permute(1, 2, 0).numpy())
    axes[1, col].set_title(f"With trigger\nPredicted: {CLASSES[tp]}",
                           fontsize=12, color="red", fontweight="bold")
    axes[1, col].add_patch(patches.Rectangle(
        (29 * SCALE - 5, 29 * SCALE - 5), 3 * SCALE + 5, 3 * SCALE + 5,
        fill=False, edgecolor="red", linewidth=2, clip_on=False))

    for row in range(2):
        axes[row, col].axis("off")

fig.suptitle("Backdoor Trigger Analysis (BadNets, 3x3 white patch, target = airplane)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/trigger_analysis.png", dpi=200)
plt.show()