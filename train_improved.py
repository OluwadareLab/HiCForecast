# train_improved.py
import argparse
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.hicforecast_modified import HiCForecastModified

class HiCDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype('float32')
        return arr[None, :, :], arr[None, :, :]  # (input, target)

def train(args):
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'

    # Load data
    files = sorted(glob.glob(args.data_path + '/*.npy'))
    split = int(0.8 * len(files))
    train_files = files[:split]
    val_files = files[split:]

    train_loader = DataLoader(HiCDataset(train_files), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(HiCDataset(val_files), batch_size=args.batch_size, shuffle=False)

    # Model
    model = HiCForecastModified(
        in_channels=args.in_channels,
        use_denoiser=True,
        use_attention=True,
        use_multimodal=False,
        mc_dropout_p=0.1,
        voxelflow_classpath=None  # use None until real voxelflow is identified
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = ((pred - yb) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {total_loss/len(train_loader):.6f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += ((pred - yb) ** 2).mean().item()

        print(f"Epoch {epoch+1}/{args.epochs} | Val Loss: {val_loss/len(val_loader):.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/synthetic")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--in_channels", type=int, default=1)
    args = parser.parse_args()

    train(args)
