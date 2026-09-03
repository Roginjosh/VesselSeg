import torch
from torch.utils.data import DataLoader

from src.dataset import VesselSegDataset
from src.unet import UNet
from src.losses import DiceLoss


def main():
    device = torch.device("cpu")

    ds = VesselSegDataset(
        csv_path="data/dataset.csv",
        img_size=256,
        augment=False,
    )

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base=64,
    ).to(device)

    bce = torch.nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    imgs, masks = next(iter(loader))

    imgs = imgs.to(device)
    masks = masks.to(device)

    print("Images:", imgs.shape)
    print("Masks:", masks.shape)

    logits = model(imgs)

    print("Logits:", logits.shape)

    loss = (
        0.5 * bce(logits, masks)
        + 0.5 * dice(logits, masks)
    )

    print("Loss:", loss.item())

    loss.backward()

    print("Backward pass: OK")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()