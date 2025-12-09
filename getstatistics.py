# eval_test.py
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.unet import UNet
from utils.losses import bce_dice
from data.dataset import ISICSegDataset   # <- or wherever your dataset lives

# ---------------- config ----------------
IMG_DIR   = Path("data/imgs")   # <-- adjust
MASK_DIR  = Path("data/masks")  # <-- adjust
CKPT_PATH = Path("runs/unet/unet_best.pt")
IMG_SIZE  = 256
BATCH_SIZE = 8
THR = 0.5
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA:", torch.cuda.get_device_name(0))

elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS accelerator")

else:
    DEVICE = torch.device("cpu")
    print("Using CPU only")



# ---------- metric helpers (logits -> scalar) ----------
@torch.no_grad()
def batch_iou_from_logits(logits: torch.Tensor,
                          target: torch.Tensor,
                          eps: float = 1e-6) -> torch.Tensor:
    """
    logits: (B,1,H,W)
    target: (B,1,H,W), float 0/1
    returns: per-sample IoU, shape (B,)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > THR).float()

    inter = (preds * target).sum(dim=(1, 2, 3))
    union = (preds + target - preds * target).sum(dim=(1, 2, 3)) + eps
    return inter / union


@torch.no_grad()
def batch_dice_from_logits(logits: torch.Tensor,
                           target: torch.Tensor,
                           eps: float = 1e-6) -> torch.Tensor:
    """
    returns: per-sample Dice, shape (B,)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > THR).float()

    inter = (preds * target).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 2 * inter / denom


# ---------------- main eval ----------------
@torch.no_grad()
def main():
    # dataset / loader
    test_ds = ISICSegDataset(
        IMG_DIR,        # image_dir
        MASK_DIR,       # mask_dir
        img_size=IMG_SIZE,
        augment=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available()

    )

    # model
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = UNet(in_ch=3, out_ch=1, base=64).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_iou = []
    all_dice = []
    all_loss = []

    for imgs, masks in tqdm(test_loader, desc="Evaluating", unit="batch"):
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        logits = model(imgs)

        loss = bce_dice(logits, masks)
        all_loss.append(loss.item())

        iou_batch  = batch_iou_from_logits(logits, masks)
        dice_batch = batch_dice_from_logits(logits, masks)

        all_iou.extend(iou_batch.cpu().numpy().tolist())
        all_dice.extend(dice_batch.cpu().numpy().tolist())

    all_iou  = np.array(all_iou)
    all_dice = np.array(all_dice)
    all_loss = np.array(all_loss)

    print("=== Test set results ===")
    print(f"Num images: {len(all_iou)}")
    print(f"Loss: mean={all_loss.mean():.4f}")
    print(f"IoU:  mean={all_iou.mean():.4f}, std={all_iou.std():.4f}, "
          f"min={all_iou.min():.4f}, max={all_iou.max():.4f}")
    print(f"Dice: mean={all_dice.mean():.4f}, std={all_dice.std():.4f}, "
          f"min={all_dice.min():.4f}, max={all_dice.max():.4f}")


if __name__ == "__main__":
    main()
