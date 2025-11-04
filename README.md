# U-Net (ISIC-style) Trainer

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# edit paths in configs/default.yaml if needed
python train.py --cfg configs/default.yaml

# quick check of the saved best checkpoint
python evaluate.py --cfg configs/default.yaml
