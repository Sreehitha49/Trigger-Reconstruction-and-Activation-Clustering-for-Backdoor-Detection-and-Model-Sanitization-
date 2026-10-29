"""
Module 1: Basic Trigger Analysis
Dataset -> Cleaning/Preprocessing -> Add Trigger -> Model Prediction -> Trigger Success Rate

Run (from project root):
    python module1_trigger_analysis.py --ckpt data/cifar10_preactresnet18/attack_result.pt

Optional:
    --patch_size 3   --target 0   --position br   --data_root data
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------- model
# (If your model.py already has PreActResNet18, you can import it instead.)
class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(PreActBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = self.avgpool(out).flatten(1)
        return self.linear(out)


def find_ckpt(path):
    """Use the given path if it exists, otherwise search under data/."""
    import glob
    if path and os.path.isfile(path):
        return path
    for pat in ["data/**/attack_result.pt", "data/**/*.pt", "data/**/*.pth"]:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            print(f"      Given path not found, using: {hits[0]}")
            return hits[0]
    raise FileNotFoundError(
        "No .pt/.pth model file found under data/. "
        "Run: Get-ChildItem -Recurse data -Include *.pt,*.pth")


def load_model(ckpt_path, device):
    ckpt_path = find_ckpt(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model = PreActResNet18(num_classes=10)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# ---------------------------------------------------------------- data
def load_cifar10_test(root):
    """Steps 1-2: load CIFAR-10 test set and preprocess to [0,1] tensors."""
    ds = torchvision.datasets.CIFAR10(root=root, train=False, download=False,
                                      transform=T.ToTensor())
    x = torch.stack([ds[i][0] for i in range(len(ds))])      # N,3,32,32
    y = torch.tensor(ds.targets)
    return x, y


def normalize(x):
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


# ---------------------------------------------------------------- trigger
def add_trigger(x, size=3, position="br", value=1.0):
    """Step 3: stamp a small square patch (default 3x3 white, bottom-right)."""
    x = x.clone()
    if position == "br":
        x[:, :, -size:, -size:] = value
    elif position == "bl":
        x[:, :, -size:, :size] = value
    elif position == "tr":
        x[:, :, :size, -size:] = value
    elif position == "tl":
        x[:, :, :size, :size] = value
    return x


# ---------------------------------------------------------------- eval
@torch.no_grad()
def predict(model, x, device, bs=500):
    """Step 4: give images to the model and collect predictions."""
    out = []
    for i in range(0, len(x), bs):
        xb = normalize(x[i:i + bs].to(device))
        out.append(model(xb).argmax(1).cpu())
    return torch.cat(out)


def clean_accuracy(pred, y):
    return (pred == y).float().mean().item()


def trigger_success_rate(pred_trig, y, target):
    """Step 6: ASR = % of non-target-class images classified as target
    after the trigger is added."""
    mask = y != target
    return (pred_trig[mask] == target).float().mean().item()


# ---------------------------------------------------------------- plots
def save_samples(x, x_trig, pred_clean, pred_trig, y, path, n=8):
    fig, ax = plt.subplots(2, n, figsize=(2 * n, 4.6))
    for i in range(n):
        ax[0, i].imshow(x[i].permute(1, 2, 0).numpy())
        ax[0, i].set_title(f"true: {CLASSES[y[i]]}\npred: {CLASSES[pred_clean[i]]}",
                           fontsize=8)
        ax[1, i].imshow(x_trig[i].permute(1, 2, 0).numpy())
        ax[1, i].set_title(f"pred: {CLASSES[pred_trig[i]]}", fontsize=8,
                           color="red" if pred_trig[i] != pred_clean[i] else "black")
        ax[0, i].axis("off")
        ax[1, i].axis("off")
    ax[0, 0].text(-0.1, 0.5, "CLEAN", rotation=90, transform=ax[0, 0].transAxes,
                  va="center", ha="right", fontsize=10, weight="bold")
    ax[1, 0].text(-0.1, 0.5, "TRIGGERED", rotation=90, transform=ax[1, 0].transAxes,
                  va="center", ha="right", fontsize=10, weight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_class_distribution(pred_clean, pred_trig, path):
    c1 = np.bincount(pred_clean.numpy(), minlength=10)
    c2 = np.bincount(pred_trig.numpy(), minlength=10)
    w = 0.4
    xs = np.arange(10)
    plt.figure(figsize=(9, 4))
    plt.bar(xs - w / 2, c1, w, label="Clean images")
    plt.bar(xs + w / 2, c2, w, label="Triggered images")
    plt.xticks(xs, CLASSES, rotation=40)
    plt.ylabel("Number of predictions")
    plt.title("Prediction distribution: clean vs triggered")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_patch_sweep(sizes, asrs, path):
    plt.figure(figsize=(5, 3.6))
    plt.plot(sizes, [a * 100 for a in asrs], marker="o")
    plt.xlabel("Trigger patch size (pixels)")
    plt.ylabel("Trigger Success Rate (%)")
    plt.title("ASR vs trigger size")
    plt.ylim(0, 105)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/cifar10_preactresnet18/attack_result.pt")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--patch_size", type=int, default=3)
    ap.add_argument("--position", default="br", choices=["br", "bl", "tr", "tl"])
    ap.add_argument("--target", type=int, default=0, help="attack target class")
    ap.add_argument("--out", default="results/module1")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Steps 1-2
    x, y = load_cifar10_test(args.data_root)
    print(f"[1-2] Loaded CIFAR-10 test set: {tuple(x.shape)}, "
          f"pixel range [{x.min():.0f}, {x.max():.0f}]")

    model = load_model(args.ckpt, device)
    print("      Model loaded: PreActResNet18")

    # Baseline on clean images
    pred_clean = predict(model, x, device)
    acc = clean_accuracy(pred_clean, y)
    print(f"      Clean accuracy (CA): {acc * 100:.2f}%")

    # Step 3
    x_trig = add_trigger(x, args.patch_size, args.position)
    print(f"[3]   Added {args.patch_size}x{args.patch_size} white patch "
          f"at '{args.position}'")

    # Steps 4-5
    pred_trig = predict(model, x_trig, device)
    to_target = (pred_trig == args.target).float().mean().item()
    print(f"[4-5] Triggered images predicted as '{CLASSES[args.target]}': "
          f"{to_target * 100:.2f}% (all images)")

    # Step 6
    asr = trigger_success_rate(pred_trig, y, args.target)
    print(f"[6]   TRIGGER SUCCESS RATE (ASR): {asr * 100:.2f}%  "
          f"(non-target images only)")

    # Extra: patch-size sweep
    sizes = list(range(1, 7))
    asrs = []
    for s in sizes:
        p = predict(model, add_trigger(x, s, args.position), device)
        asrs.append(trigger_success_rate(p, y, args.target))
    print("\nPatch size sweep:")
    for s, a in zip(sizes, asrs):
        print(f"   {s}x{s}: ASR = {a * 100:.2f}%")

    # Save outputs
    save_samples(x, x_trig, pred_clean, pred_trig, y, f"{args.out}/samples.png")
    save_class_distribution(pred_clean, pred_trig, f"{args.out}/distribution.png")
    save_patch_sweep(sizes, asrs, f"{args.out}/patch_sweep.png")
    with open(f"{args.out}/results.json", "w") as f:
        json.dump({
            "clean_accuracy": acc,
            "trigger_success_rate": asr,
            "target_class": CLASSES[args.target],
            "patch_size": args.patch_size,
            "position": args.position,
            "patch_sweep": {str(s): a for s, a in zip(sizes, asrs)},
        }, f, indent=2)
    print(f"\nSaved plots and results.json to: {args.out}/")


if __name__ == "__main__":
    main()