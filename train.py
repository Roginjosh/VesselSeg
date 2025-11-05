import argparse, yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from data.dataset import ISICSegDataset
from models.unet import UNet
from engine.train_eval import run_epoch
from utils.seed import seed_everything

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    seed_everything(cfg.get("seed", 42))

    # ----- data -----
    paths  = cfg["paths"]
    data_c = cfg["data"]

    dataset = ISICSegDataset(
        paths["images_dir"],
        paths["masks_dir"],
        img_size=data_c["img_size"],
        augment=data_c["augment"]
    )

    val_len   = max(1, int(len(dataset) * data_c["val_frac"]))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len],
                                      generator=torch.Generator().manual_seed(cfg.get("seed", 42)))

    train_loader = DataLoader(
        train_set,
        batch_size=data_c["batch_size"],
        shuffle=True,
        num_workers=data_c["num_workers"],
        pin_memory=data_c["pin_memory"]
    )
    val_loader = DataLoader(
        val_set,
        batch_size=data_c["batch_size"],
        shuffle=False,
        num_workers=data_c["num_workers"],
        pin_memory=data_c["pin_memory"]
    )

    # quick peek (first batch)
    imgs, masks = next(iter(train_loader))
    k = min(3, imgs.size(0))
    plt.figure(figsize=(10, 3*k))
    for i in range(k):
        img = np.transpose(imgs[i].numpy(), (1,2,0))
        msk = masks[i,0].numpy()
        plt.subplot(k,2,2*i+1); plt.imshow(img); plt.title('Image'); plt.axis('off')
        plt.subplot(k,2,2*i+2); plt.imshow(msk, cmap='gray'); plt.title('Mask'); plt.axis('off')
    Path(paths["out_dir"]).mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(Path(paths["out_dir"]) / "peek.png"); plt.close()

    # ----- model/optim -----
    model_c = cfg["model"]
    model = UNet(in_ch=model_c["in_ch"], out_ch=model_c["out_ch"], base=model_c["base"])

    train_c = cfg["train"]


    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Using device:", device)
    model.to(device)

    opt_c = cfg["optim"]
    if opt_c["name"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_c["lr"], weight_decay=opt_c["weight_decay"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt_c["lr"], weight_decay=opt_c["weight_decay"])

    # ----- training loop -----
    best_val = -1.0
    best_path = Path(paths["out_dir"]) / train_c["best_ckpt"]
    epochs = int(train_c["epochs"])

    for e in range(1, epochs+1):
        tr_l, tr_d, tr_i = run_epoch(model, train_loader, optimizer, device, train=True, amp=train_c["amp"])
        va_l, va_d, va_i = run_epoch(model,  val_loader,  optimizer, device, train=False, amp=train_c["amp"])

        if va_d > best_val:
            best_val = va_d
            torch.save({'model': model.state_dict(), 'img_size': data_c['img_size']}, best_path)

        print(f"E{e:02d} | train: loss {tr_l:.4f} dice {tr_d:.4f} iou {tr_i:.4f}  ||  "
              f"val: loss {va_l:.4f} dice {va_d:.4f} iou {va_i:.4f}")

    print("Best model saved to:", str(best_path))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    args = ap.parse_args()
    main(args.cfg)
