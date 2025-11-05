# 🧬 Deep Learning Final Project — U-Net Vignetting Filter

This repository contains a PyTorch implementation of a **U-Net** architecture for removing or segmenting **vignetting artifacts** in dermoscopic images.  
The project uses the ISIC 2019 dataset and demonstrates training, evaluation, and visualization of segmentation results.

---

## 🚀 Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/PixelatedYeti/DeepLearningFinalProject.git
cd DeepLearningFinalProject
python -m venv .venv
source .venv/bin/activate   # (Windows: .\.venv\Scripts\Activate.ps1)
pip install -r requirements.txt



## 🔽 Download Pretrained Model
# Create output folder if needed
mkdir -p runs/unet

# Download checkpoint (~118 MB)
curl -L -o runs/unet/unet_best.pt \
  https://huggingface.co/PixelatedYeti/DeepLearningFinalProject/resolve/main/unet_best.pt

## 📊 Evaluate
python evaluate.py --ckpt runs/unet/unet_best.pt


## 🧩 Train from scratch
python train.py --cfg configs/default.yaml

## File Structure
DeepLearningFinalProject/
├── configs/               # YAML configuration files
├── data/                  # Input images and masks
├── engine/                # Training and evaluation logic
├── models/                # Model definitions (UNet)
├── utils/                 # Metrics and helper functions
├── runs/                  # Model checkpoints, logs, outputs
├── train.py               # Main training script
├── evaluate.py            # Quantitative evaluation
├── visualize_one.py       # Single-image overlay visualization
└── requirements.txt       # Dependencies
