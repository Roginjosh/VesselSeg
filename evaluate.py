# visualize_one.py
import torch, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from models.unet import UNet

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
    if mask_path.exists():
        mask = Image.open(mask_path).convert("L").resize((img_size, img_size))
        mask_arr = np.array(mask) > 127
    return img_t, img_arr, mask_arr, mask_path if mask_path.exists() else None

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
    img_t, img_np, mask_np, mask_path = load_image_mask(img_path, MASKS_DIR, IMG_SIZE)

    with torch.no_grad():
        logits = model(img_t.to(device))
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = probs > THR

    # ---- figure ----
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
 
    if mask_np is not None:
        axs[0].imshow(mask_np, cmap="gray")
        axs[0].set_title(f"Ground Truth\n({mask_path.name})")
    else:
        axs[0].text(0.5, 0.5, "No mask found", ha="center", va="center", fontsize=12)
        axs[0].set_title("Ground Truth (missing)")
    axs[0].axis("off")

    axs[1].imshow(probs, cmap="viridis")
    axs[1].set_title("Predicted Probability")
    axs[1].axis("off")

    axs[2].imshow(img_np)
    axs[2].imshow(pred, cmap="Reds", alpha=0.4)
    axs[2].set_title(f"Overlay (thr={THR})")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
