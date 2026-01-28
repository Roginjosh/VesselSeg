import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import ISICKeySegDataset
from unet import UNet

def to_display_img(x_chw: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_chw.device)[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225], device=x_chw.device)[:, None, None]
    x = (x_chw * std + mean).clamp(0, 1)
    return x.permute(1, 2, 0).cpu().numpy()

def bottom_title(ax, text):
    # Place text inside the axes at the bottom center
    ax.text(
        0.5, 0.02, text,                 # x,y in axes coords
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=10
    )


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps=1e-7):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2 * inter + eps) / (pred.sum() + gt.sum() + eps)
    iou  = (inter + eps) / (union + eps)
    return dice, iou

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="unet_best.pt")
    ap.add_argument("--img_dir", default="data/imgs")
    ap.add_argument("--mask_dir", default="data/masks")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="List of thresholds to visualize"
    )
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    ds = ISICKeySegDataset(args.img_dir, args.mask_dir, img_size=args.size, augment=False)

    model = UNet(in_channels=3, out_channels=1, base=64).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    x, y = ds[args.idx]
    img = to_display_img(x)
    gt  = y[0].cpu().numpy().astype(np.uint8)

    logits = model(x.unsqueeze(0).to(device))[0, 0]
    prob = torch.sigmoid(logits).cpu().numpy()

    thresholds = args.thresholds[:5]  # enforce max 5
    n = len(thresholds)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12, 8),
        constrained_layout=True
    )

    # Flatten for easy indexing
    axes = axes.flatten()

    # --- Panel 0: original image ---
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")

    # --- Threshold panels ---
    for i, thr in enumerate(thresholds):
        pred = (prob >= thr).astype(np.uint8)

        tp = (pred == 1) & (gt == 1)
        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)

        overlay = np.zeros((*gt.shape, 3), dtype=np.float32)
        overlay[tp, 1] = 1.0  # green
        overlay[fp, 0] = 1.0  # red
        overlay[fn, 2] = 1.0  # blue

        d, iou = dice_iou(pred, gt)

        ax = axes[i + 1]
        ax.imshow(img)
        ax.imshow(overlay, alpha=args.alpha)
        title_text = f"thr={thr:.2f}\nDice={d:.3f} IoU={iou:.3f}"

        if i + 1 in (1, 2):      # top row panels
            ax.set_title(title_text)
        else:                    # bottom row panels
            bottom_title(ax, title_text)
        ax.axis("off")

        print(f"thr={thr:.2f}  Dice={d:.4f}  IoU={iou:.4f}")

    # Hide any unused panels (just in case)
    for j in range(n + 1, 6):
        axes[j].axis("off")

    out_path = f"demo/overlay_idx_{args.idx}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


    plt.show()

if __name__ == "__main__":
    main()
