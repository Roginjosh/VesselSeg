import os, random, numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

IMAGES_DIR = './data/imgs'
MASKS_DIR = './data/masks'

class ISICSegDataset(Dataset):
    """
    Loads RGB image and single-channel binary mask.
    - Resizes to IMG_SIZE
    - Optional simple, synchronized augmentations
    Expects masks named: <image_stem>_mvig.tif[f]
    """
    def __init__(self, image_dir, mask_dir, img_size=256, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.paths = sorted([p for p in self.image_dir.iterdir()
                             if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'}])
        self.img_size = img_size
        self.augment = augment

        self.resize_img  = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        base = img_path.stem
        mpath = self.mask_dir / f'{base}_mvig.tif'
        if not mpath.exists():
            mpath = self.mask_dir / f'{base}_mvig.tiff'
        if not mpath.exists():
            raise FileNotFoundError(f"Mask not found for {base}")

        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mpath)
        if mask.mode != 'L':
            mask = mask.convert('L')

        img  = self.resize_img(img)
        mask = self.resize_mask(mask)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img  = TF.hflip(img);  mask = TF.hflip(mask)
            if torch.rand(1).item() < 0.5:
                img  = TF.vflip(img);  mask = TF.vflip(mask)

        img_t  = TF.to_tensor(img)          # (3,H,W), [0,1]
        mask_t = TF.to_tensor(mask)         # (1,H,W), [0,1]
        mask_t = (mask_t > 0).float()       # binarize
        return img_t, mask_t

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=64):
        super().__init__()
        self.c1 = conv_block(in_ch, base);   self.p1 = nn.MaxPool2d(2)
        self.c2 = conv_block(base, base*2);  self.p2 = nn.MaxPool2d(2)
        self.c3 = conv_block(base*2, base*4);self.p3 = nn.MaxPool2d(2)
        self.c4 = conv_block(base*4, base*8);self.p4 = nn.MaxPool2d(2)
        self.c5 = conv_block(base*8, base*16)

        self.u6 = nn.ConvTranspose2d(base*16, base*8, 2, 2); self.c6 = conv_block(base*16, base*8)
        self.u7 = nn.ConvTranspose2d(base*8,  base*4, 2, 2); self.c7 = conv_block(base*8,  base*4)
        self.u8 = nn.ConvTranspose2d(base*4,  base*2, 2, 2); self.c8 = conv_block(base*4,  base*2)
        self.u9 = nn.ConvTranspose2d(base*2,  base,   2, 2); self.c9 = conv_block(base*2,  base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.p1(c1))
        c3 = self.c3(self.p2(c2))
        c4 = self.c4(self.p3(c3))
        c5 = self.c5(self.p4(c4))

        x  = self.u6(c5); x = torch.cat([x, c4], 1); x = self.c6(x)
        x  = self.u7(x);  x = torch.cat([x, c3], 1); x = self.c7(x)
        x  = self.u8(x);  x = torch.cat([x, c2], 1); x = self.c8(x)
        x  = self.u9(x);  x = torch.cat([x, c1], 1); x = self.c9(x)
        return self.out(x)  # logits

# ---- losses/metrics (kept at module scope so testing.py can import) ----
bce = nn.BCEWithLogitsLoss()

def dice_coef(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * target).sum((1,2,3))
    denom = preds.sum((1,2,3)) + target.sum((1,2,3)) + eps
    return ((2*inter + eps) / denom).mean()

def iou_coef(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * target).sum((1,2,3))
    union = (preds + target - preds*target).sum((1,2,3)) + eps
    return (inter / union).mean()

def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum((1,2,3))
    denom = probs.sum((1,2,3)) + target.sum((1,2,3)) + eps
    return 1 - ((2*inter + eps) / denom).mean()

def bce_dice(logits, target):
    return bce(logits, target) + dice_loss(logits, target)

# ================== TRAINING (guarded) ==================
if __name__ == "__main__":
    # Build dataset + split
    IMG_SIZE   = 256        # use 256 for speed
    VAL_FRAC   = 0.3
    dataset    = ISICSegDataset(IMAGES_DIR, MASKS_DIR, img_size=IMG_SIZE, augment=True)

    val_len  = max(1, int(len(dataset)*VAL_FRAC))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len],
                                      generator=torch.Generator().manual_seed(42))

    BATCH_SIZE = 8
    NUM_WORKERS = 0
    PIN_MEMORY   = False

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # quick peek
    imgs, masks = next(iter(train_loader))
    k = min(3, imgs.size(0))
    plt.figure(figsize=(10, 3*k))
    for i in range(k):
        img  = np.transpose(imgs[i].numpy(), (1,2,0))
        msk  = masks[i,0].numpy()
        plt.subplot(k,2,2*i+1); plt.imshow(img); plt.title('Image'); plt.axis('off')
        plt.subplot(k,2,2*i+2); plt.imshow(msk, cmap='gray'); plt.title('Mask'); plt.axis('off')
    plt.tight_layout(); plt.show()

    # model/opt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_ch=3, out_ch=1, base=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    def run_epoch(loader, train=True):
        model.train(train)
        tot_loss = tot_dice = tot_iou = 0.0
        n = 0
        loop = tqdm(loader, total=len(loader), leave=False)
        loop.set_description("Train" if train else "Val")
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = bce_dice(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            with torch.no_grad():
                dice = dice_coef(logits, masks).item()
                iou  = iou_coef(logits, masks).item()
            bs = imgs.size(0)
            tot_loss += loss.item() * bs
            tot_dice += dice * bs
            tot_iou  += iou  * bs
            n += bs
            loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.3f}", iou=f"{iou:.3f}")
        return tot_loss/n, tot_dice/n, tot_iou/n

    # train briefly
    
    EPOCHS = 10
    best_val = -1
    best_path = './unet_best.pt'
    for e in range(1, EPOCHS+1):
        tr_l, tr_d, tr_i = run_epoch(train_loader, True)
        va_l, va_d, va_i = run_epoch(val_loader,   False)
        if va_d > best_val:
            best_val = va_d
            torch.save({'model': model.state_dict(),
                        'img_size': IMG_SIZE}, best_path)
        print(f'E{e:02d} | train: loss {tr_l:.4f} dice {tr_d:.4f} iou {tr_i:.4f}  ||  '
              f'val: loss {va_l:.4f} dice {va_d:.4f} iou {va_i:.4f}')
    print("Best model saved to:", best_path)
