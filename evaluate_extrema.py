# evaluate_extrema.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from models.unet import UNet
from utils.losses import bce_dice
from utils.metrics import dice_coef, iou_coef

# ---------- config ----------
IMAGES_DIR = Path("data/imgs")
MASKS_DIR  = Path("data/masks")
CKPT_PATH  = Path("runs/unet/unet_best.pt")
IMG_SIZE   = 256
THR        = 0.5

OUT_DIR = Path("runs/unet/extrema")
OUT_DIR_WORST = OUT_DIR / "worst"
OUT_DIR_BEST  = OUT_DIR / "best"

# These are YOUR lists from eval_test.py:
WORST_10 = [
    "ISIC_0071478.jpg",
    "ISIC_0070763.jpg",
    "ISIC_0054033.jpg",
    "ISIC_0034858.jpg",
    "ISIC_0053995.jpg",
    "ISIC_0071025.jpg",
    "ISIC_0071234.jpg",
    "ISIC_0054579.jpg",
    "ISIC_0071310.jpg",
    "ISIC_0071390.jpg"
]

BEST_10 = [
    "ISIC_0071219.jpg",
    "ISIC_0071221.jpg",
    "ISIC_0054095.jpg",
    "ISIC_0071469.jpg",
    "ISIC_0054195.jpg",
    "ISIC_0071259.jpg",
    "ISIC_0072411.jpg",
    "ISIC_0071658.jpg",
    "ISIC_0071370.jpg",
    "ISIC_0071784.jpg"
]


# ---------- load image+mask like evaluate.py ----------
def load_image_mask(img_path: Path, mask_dir: Path, img_size: int):
    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    base = img_path.stem
    mask_path = mask_dir / f"{base}_mvig.tif"
    if not mask_path.exists():
        mask_path = mask_dir / f"{base}_mvig.tiff"

    mask_np = None
    mask_t = None
    if mask_path.exists():
        mask = Image.open(mask_path).convert("L").resize((img_size, img_size))
        mask_np = np.array(mask) > 127
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    return img_t, img_np, mask_np, mask_t, mask_path


# ---------- viz & save ----------
@torch.no_grad()
def visualize_and_save(model, fname, out_path):
    img_path = IMAGES_DIR / fname
    img_t, img_np, mask_np, mask_t, mask_path = load_image_mask(img_path, MASKS_DIR, IMG_SIZE)

    logits = model(img_t.to(DEVICE))
    probs = torch.sigmoid(logits)
    prob_np = probs[0, 0].cpu().numpy()
    pred = prob_np > THR

    loss_val = dice_val = iou_val = None
    if mask_t is not None:
        loss_val = bce_dice(logits, mask_t.to(DEVICE)).item()
        dice_val = dice_coef(logits, mask_t.to(DEVICE)).item()
        iou_val = iou_coef(logits, mask_t.to(DEVICE)).item()

    # Error map
    err_map = None
    if mask_np is not None:
        gt = mask_np.astype(bool)
        pr = pred.astype(bool)

        tp = gt & pr
        fp = ~gt & pr
        fn = gt & ~pr

        err_map = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        err_map[tp] = [0.0, 1.0, 0.0]
        err_map[fp] = [1.0, 0.0, 0.0]
        err_map[fn] = [0.0, 0.0, 1.0]

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    title = f"{fname}"
    if iou_val is not None:
        title += f" | IoU={iou_val:.4f}, Dice={dice_val:.4f}, Loss={loss_val:.4f}"

    fig.suptitle(title, fontsize=14)

    axs[0, 0].imshow(img_np); axs[0, 0].set_title("Original"); axs[0, 0].axis("off")

    if mask_np is not None:
        axs[0, 1].imshow(mask_np, cmap="gray")
        axs[0, 1].set_title("Ground Truth")
    else:
        axs[0, 1].text(0.5, 0.5, "No mask", ha="center")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(prob_np, cmap="viridis")
    axs[1, 0].set_title("Predicted Probability")
    axs[1, 0].axis("off")

    if err_map is not None:
        axs[1, 1].imshow(img_np)
        axs[1, 1].imshow(err_map, alpha=0.6)
        axs[1, 1].set_title("Error Map\nGreen=TP Red=FP Blue=FN")
    axs[1, 1].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


# ---------- main ----------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

@torch.no_grad()
def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = UNet(in_ch=3, out_ch=1, base=64).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Worst 10
    for fname in WORST_10:
        out_path = OUT_DIR_WORST / f"worst_{fname.replace('.jpg','')}.png"
        visualize_and_save(model, fname, out_path)

    # Best 10
    for fname in BEST_10:
        out_path = OUT_DIR_BEST / f"best_{fname.replace('.jpg','')}.png"
        visualize_and_save(model, fname, out_path)


if __name__ == "__main__":
    main()
