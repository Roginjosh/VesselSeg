# visualize_one.py
import torch, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from models.unet import UNet
from utils.losses import bce_dice           # <-- your loss
from utils.metrics import dice_coef, iou_coef  # <-- your metrics

# ---- parameters ----
IMAGES_DIR = Path("data/imgs")
MASKS_DIR  = Path("data/masks")
CKPT_PATH  = Path("runs/unet/unet_best.pt")
IMG_SIZE   = 256
THR        = 0.5

def load_image_mask(img_path: Path, mask_dir: Path, img_size: int):
    # load and resize RGB
    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    img_arr = np.array(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    # find matching mask
    base = img_path.stem
    mask_path = mask_dir / f"{base}_mvig.tif"
    if not mask_path.exists():
        mask_path = mask_dir / f"{base}_mvig.tiff"

    mask_arr = None
    mask_t = None
    if mask_path.exists():
        mask = Image.open(mask_path).convert("L").resize((img_size, img_size))
        mask_arr = np.array(mask) > 127    # bool mask
        # tensor mask: (1,1,H,W) float
        mask_t = torch.from_numpy(mask_arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    return img_t, img_arr, mask_arr, mask_t, mask_path if mask_path.exists() else None

def main():
    # pick a random image
    img_files = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
    if not img_files:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")
    img_path = random.choice(img_files)
    print(f"[INFO] Using random image: {img_path.name}")

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_ch=3, out_ch=1, base=64).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- data ----
    img_t, img_np, mask_np, mask_t, mask_path = load_image_mask(img_path, MASKS_DIR, IMG_SIZE)

    loss_val = dice_val = iou_val = None

    with torch.no_grad():
        logits = model(img_t.to(device))          # (1,1,H,W)
        probs  = torch.sigmoid(logits)           # (1,1,H,W)
        prob_np = probs[0, 0].cpu().numpy()      # (H,W)
        pred   = prob_np > THR                   # bool (H,W)

        if mask_t is not None:
            # ---- compute loss + metrics ----
            loss_val = bce_dice(logits, mask_t.to(device)).item()
            dice_val = dice_coef(logits, mask_t.to(device)).item()
            iou_val  = iou_coef(logits, mask_t.to(device)).item()

            print(f"[INFO] Loss = {loss_val:.4f}")
            print(f"[INFO] Dice = {dice_val:.4f}")
            print(f"[INFO] IoU  = {iou_val:.4f}")
        else:
            print("[INFO] No mask found; skipping loss/metrics")

    # ---- build error map (TP/FP/FN) if mask exists ----
    err_map = None
    if mask_np is not None:
        gt = mask_np.astype(bool)
        pr = pred.astype(bool)

        tp = gt & pr          # lesion, predicted lesion
        fp = ~gt & pr         # background, predicted lesion
        fn = gt & ~pr         # lesion, predicted background

        iou_manual = tp_count / (tp_count + fp_count + fn_count + 1e-6)
        print(f"Manual IoU: {iou_manual:.6f}")
            

        err_map = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        err_map[tp] = [0.0, 1.0, 0.0]   # green
        err_map[fp] = [1.0, 0.0, 0.0]   # red
        err_map[fn] = [0.0, 0.0, 1.0]   # blue

    # ---- figure ----
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # global title with metrics
    if loss_val is not None:
        fig.suptitle(
            f"{img_path.name}  |  Loss={loss_val:.4f}, Dice={dice_val:.4f}, IoU={iou_val:.4f}",
            fontsize=14
        )
    else:
        fig.suptitle(f"{img_path.name}  |  Loss/Dice/IoU: N/A (no mask)", fontsize=14)

    # (0,0) original image
    axs[0, 0].imshow(img_np)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    # (0,1) ground truth
    if mask_np is not None:
        axs[0, 1].imshow(mask_np, cmap="gray")
        axs[0, 1].set_title(f"Ground Truth\n({mask_path.name})")
    else:
        axs[0, 1].text(0.5, 0.5, "No mask found", ha="center", va="center", fontsize=12)
        axs[0, 1].set_title("Ground Truth (missing)")
    axs[0, 1].axis("off")

    # (1,0) predicted probability map
    axs[1, 0].imshow(prob_np, cmap="viridis")
    axs[1, 0].set_title("Predicted Probability")
    axs[1, 0].axis("off")

    # (1,1) overlay or error map
    if err_map is not None:
        axs[1, 1].imshow(img_np)
        axs[1, 1].imshow(err_map, alpha=0.6)
        axs[1, 1].set_title("Error Map\nGreen=TP, Red=FP, Blue=FN")
    else:
        axs[1, 1].imshow(img_np)
        axs[1, 1].set_title(f"Overlay (thr={THR})")
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
