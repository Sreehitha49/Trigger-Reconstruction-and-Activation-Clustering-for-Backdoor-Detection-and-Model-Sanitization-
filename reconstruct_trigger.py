import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import PreActResNet18

torch.manual_seed(0)

TARGET = 0          # airplane
STEPS = 300         # lower to 100 for a quick test
LAMBDA = 0.005      # penalty that keeps the mask small
BATCH = 128

MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
norm = lambda x: (x - MEAN) / STD

# Load the suspicious model and freeze it
path = "data/cifar10_preactresnet18_badnet_0_01/attack_result.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
model = PreActResNet18(num_classes=10)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Small clean dataset: non-target images only
dataset = torchvision.datasets.CIFAR10(root="data", train=False,
                                       download=False,
                                       transform=transforms.ToTensor())
imgs = []
for i in range(2000):
    x, y = dataset[i]
    if y != TARGET:
        imgs.append(x)
imgs = torch.stack(imgs)
opt_imgs, eval_imgs = imgs[:1000], imgs[1000:]   # optimize on one part, test on the rest
print("Images for optimization:", len(opt_imgs), "| held-out:", len(eval_imgs))

# Learnable trigger: mask (where) and pattern (what)
mask_logits = nn.Parameter(torch.full((1, 1, 32, 32), -3.0))
pattern_logits = nn.Parameter(torch.zeros(1, 3, 32, 32))
optimizer = torch.optim.Adam([mask_logits, pattern_logits], lr=0.1)
ce_loss = nn.CrossEntropyLoss()

for step in range(STEPS + 1):
    idx = torch.randperm(len(opt_imgs))[:BATCH]
    x = opt_imgs[idx]
    mask = torch.sigmoid(mask_logits)
    pattern = torch.sigmoid(pattern_logits)

    x_adv = x * (1 - mask) + pattern * mask
    out = model(norm(x_adv))
    target = torch.full((len(x),), TARGET, dtype=torch.long)

    loss = ce_loss(out, target) + LAMBDA * mask.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        acc = (out.argmax(1) == TARGET).float().mean().item() * 100
        print(f"Step {step:3d} | loss {loss.item():.3f} | "
              f"mask size {mask.sum().item():.1f} | target rate {acc:.1f}%")

# Evaluate on held-out images
# Evaluate on held-out images
with torch.no_grad():
    mask = torch.sigmoid(mask_logits)
    pattern = torch.sigmoid(pattern_logits)
    x_adv = eval_imgs * (1 - mask) + pattern * mask
    pred_adv = model(norm(x_adv)).argmax(1)
    pred_clean = model(norm(eval_imgs)).argmax(1)
    success = (pred_adv == TARGET).float().mean().item() * 100
    in_region = mask[0, 0, 29:32, 29:32].sum().item() / mask.sum().item() * 100

print("\n--- Reconstruction result ---")
print(f"Mask size (L1 norm): {mask.sum().item():.1f} of 1024 pixels")
print(f"Reconstructed trigger success on held-out images: {success:.2f}%")
print(f"Share of mask inside the true trigger corner (3x3): {in_region:.1f}%")

os.makedirs("results", exist_ok=True)
torch.save({
    "mask": mask, "pattern": pattern,
    "x_clean": eval_imgs[:200], "x_adv": x_adv[:200],
    "pred_clean": pred_clean[:200], "pred_adv": pred_adv[:200],
    "success": success, "in_region": in_region,
}, "results/reconstruction.pt")
print("Saved results/reconstruction.pt -> now run: python plot_reconstruction.py")