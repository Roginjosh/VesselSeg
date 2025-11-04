import argparse, yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from data.dataset import ISICSegDataset
from models.unet import UNet
from utils.metrics import dice_coef, iou_coef

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    paths, data_c, model_c, train_c = cfg["paths"], cfg["data"], cfg["model"], cfg["train"]

    ds = ISICSegDataset(paths["images_dir"], paths["masks_dir"], img_size=data_c["img_size"], augment=False)
    loader = DataLoader(ds, batch_size=data_c["batch_size"], shuffle=False,
                        num_workers=data_c["num_workers"], pin_memory=data_c["pin_memory"])

    device = torch.device(train_c["device"] if train_c["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    model = UNet(**model_c).to(device)

    ckpt_path = Path(paths["out_dir"]) / train_c["best_ckpt"]
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    imgs, masks = next(iter(loader))
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad():
        logits = model(imgs)
        d = dice_coef(logits, masks).item()
        j = iou_coef(logits, masks).item()
    print(f"Eval on first batch -> Dice: {d:.4f} | IoU: {j:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    args = ap.parse_args()
    main(args.cfg)
