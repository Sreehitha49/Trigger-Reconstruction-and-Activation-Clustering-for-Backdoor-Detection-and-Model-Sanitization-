"""
Compare the KNOWN trigger (3x3 white patch, bottom-right) with the
RECONSTRUCTED trigger mask saved by reconstruct_trigger.py.

Run:  python compare_triggers.py
Needs: results/reconstruction.pt  (created by reconstruct_trigger.py)
"""
import torch
import matplotlib.pyplot as plt

KNOWN_ASR = 74.62   # from calculate_asr.py (known 3x3 trigger)
K = 24              # number of strongest reconstructed pixels to compare

d = torch.load("results/reconstruction.pt", weights_only=False)
mask = d["mask"][0, 0]                       # 32 x 32, values 0..1
rec_success = d["success"]

# Known trigger: 3x3 patch in the bottom-right corner
known = torch.zeros(32, 32)
known[29:32, 29:32] = 1.0

# Strongest K pixels of the reconstructed mask
top = torch.topk(mask.flatten(), K).indices
top_mask = torch.zeros(32 * 32)
top_mask[top] = 1.0
top_mask = top_mask.view(32, 32)

inside = int((top_mask * known).sum().item())
covered = int((top_mask * known).sum().item())   # true-trigger pixels hit

# Overlay: red = reconstructed, cyan = known, white = both
rgb = torch.zeros(32, 32, 3)
rgb[..., 0] = top_mask
rgb[..., 1] = known
rgb[..., 2] = known

print("\n=== Known vs reconstructed trigger ===")
print(f"{'':32s}{'Known':>12s}{'Reconstructed':>16s}")
print(f"{'Trigger size (pixels)':32s}{9:>12d}{mask.sum().item():>16.1f}")
print(f"{'Attack success rate (%)':32s}{KNOWN_ASR:>12.2f}{rec_success:>16.2f}")
print(f"Strongest {K} reconstructed pixels inside the true 3x3 corner: "
      f"{inside} of {K}")
print(f"True-trigger pixels hit by those {K}: {covered} of 9")
print(f"Share of mask weight inside true corner: {d['in_region']:.1f}%")

fig, ax = plt.subplots(1, 3, figsize=(15, 5.8), dpi=110,
                       constrained_layout=True)

ax[0].imshow(known, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[0].set_title(f"KNOWN trigger\n3x3 white patch (9 pixels)\n"
                f"ASR = {KNOWN_ASR:.2f}%", fontsize=12)

im = ax[1].imshow(mask.numpy(), cmap="inferno", vmin=0,
                  vmax=mask.max().item(), interpolation="nearest")
ax[1].set_title(f"RECONSTRUCTED mask\n(found without knowing the answer)\n"
                f"success = {rec_success:.2f}%", fontsize=12)
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)

ax[2].imshow(rgb.numpy(), interpolation="nearest")
ax[2].set_title(f"OVERLAY\nred = reconstructed (top {K}), cyan = known\n"
                f"white = overlap ({inside} pixels)", fontsize=12)

for a in ax:
    a.set_xticks([])
    a.set_yticks([])

fig.suptitle("Known Trigger vs Reconstructed Trigger", fontsize=15,
             fontweight="bold")
plt.savefig("results/compare_known_vs_reconstructed.png", dpi=200,
            bbox_inches="tight")
print("\nSaved: results/compare_known_vs_reconstructed.png")
plt.show()