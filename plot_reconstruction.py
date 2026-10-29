import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
S = 10   # display only: enlarge 32x32 -> 320x320
K = 24   # number of strongest mask pixels to highlight

d = torch.load("results/reconstruction.pt", weights_only=False)
mask = d["mask"][0, 0]
ok = ((d["pred_clean"] != 0) & (d["pred_adv"] == 0)).nonzero()
i = ok[0].item() if len(ok) else 0

def up(img):
    return F.interpolate(img.unsqueeze(0), scale_factor=S, mode="bicubic",
                         align_corners=False).squeeze(0).clamp(0, 1)

clean = up(d["x_clean"][i]).permute(1, 2, 0).numpy()
adv = up(d["x_adv"][i]).permute(1, 2, 0).numpy()

# strongest K mask pixels
top = torch.topk(mask.flatten(), K).indices
rows, cols = top // 32, top % 32

fig, ax = plt.subplots(1, 4, figsize=(18, 5.4), dpi=110,
                       constrained_layout=True)

ax[0].imshow(clean)
ax[0].set_title(f"Clean image\nPredicted: {CLASSES[d['pred_clean'][i]]}", fontsize=12)

im = ax[1].imshow(mask.numpy(), cmap="inferno", vmin=0, vmax=mask.max().item(),
                  interpolation="nearest")
ax[1].set_title("Reconstructed mask\n(brighter = stronger change)", fontsize=12)
ax[1].add_patch(patches.Rectangle((28.5, 28.5), 3, 3, fill=False,
                                  edgecolor="cyan", linewidth=2))
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)

ax[2].imshow(clean)
for r, c in zip(rows.tolist(), cols.tolist()):
    ax[2].add_patch(patches.Rectangle((c * S, r * S), S, S, fill=False,
                                      edgecolor="red", linewidth=1.5))
ax[2].add_patch(patches.Rectangle((29 * S, 29 * S), 3 * S, 3 * S, fill=False,
                                  edgecolor="cyan", linewidth=2.5))
ax[2].set_title(f"Strongest {K} mask pixels (red)\ntrue trigger location (cyan)",
                fontsize=12)

ax[3].imshow(adv)
ax[3].set_title(f"With reconstructed trigger\nPredicted: {CLASSES[d['pred_adv'][i]]}",
                fontsize=12, color="red", fontweight="bold")

for a in ax:
    a.set_xticks([])
    a.set_yticks([])

fig.suptitle("Trigger Reconstruction on the Backdoored Model", fontsize=15,
             fontweight="bold")
fig.text(0.5, -0.03,
         f"Mask size: {mask.sum().item():.1f} of 1024 pixels   |   "
         f"Success on held-out images: {d['success']:.2f}%   |   "
         f"Overlap with true trigger corner: {d['in_region']:.1f}%",
         ha="center", fontsize=12)

plt.savefig("results/reconstructed_trigger_clean.png", dpi=200,
            bbox_inches="tight")
plt.show()