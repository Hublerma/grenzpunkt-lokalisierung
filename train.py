"""Stufe 1: U-Net sagt aus einem 256px-Ausschnitt eine Heatmap der Grenzpunktsymbole voraus.
Aufbau nach den PyTorch-Tutorials (Datasets & DataLoaders, Quickstart) und dem Quickstart von segmentation_models_pytorch."""
import glob
import random

import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

DATA = "data/ausschnitte_256px_600dpi"  # erstellt mit aufbereitung/ausschnitte_erstellen.py
SIGMA = 2.0  # px, Breite der Gaussglocke
BATCH_SIZE = 16
EPOCHS = 20
device = torch.device("mps")


def heatmap(punkte, h, w):
    """Gaussglocke pro Punkt (Mitte von Pixel i liegt bei i + 0.5), ueberlappende Glocken per Maximum."""
    x = torch.arange(w) + 0.5
    y = torch.arange(h) + 0.5
    hm = torch.zeros(1, h, w)
    for px, py in punkte:
        hm = torch.maximum(hm, torch.exp(-((x[None, :] - px) ** 2 + (y[:, None] - py) ** 2) / (2 * SIGMA**2)))
    return hm


class PunktDataset(Dataset):
    def __init__(self, split, augment=False):
        df = pd.concat(pd.read_csv(f) for f in glob.glob(f"{DATA}/punkte/*.csv"))
        df = df[df["split"] == split]
        self.bilder = df["bild"].unique()  # vor dropna, damit Ausschnitte ohne Punkt bleiben
        df = df.dropna(subset=["x"])
        self.punkte = {b: g[["x", "y"]].values for b, g in df.groupby("bild")}
        self.augment = augment
        self.jitter = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet

    def __len__(self):
        return len(self.bilder)

    def __getitem__(self, idx):
        bild = self.bilder[idx]
        image = decode_image(f"{DATA}/bilder/{bild}.png", mode=ImageReadMode.RGB)
        image = v2.functional.to_dtype(image, torch.float32, scale=True)
        label = heatmap(self.punkte.get(bild, []), *image.shape[1:])
        if self.augment:  # Heatmap wird mitgedreht/-gespiegelt, statt die Punkte zu transformieren
            image = self.jitter(image)
            k = random.randint(0, 3)
            image, label = torch.rot90(image, k, (1, 2)), torch.rot90(label, k, (1, 2))
            if random.random() < 0.5:
                image, label = image.flip(2), label.flip(2)
        return self.normalize(image), label


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_loss += loss_fn(model(X), y).item()
    test_loss /= num_batches
    print(f"Val loss: {test_loss:>8f}\n")
    return test_loss


if __name__ == "__main__":  # noetig fuer num_workers > 0 auf macOS
    train_dataloader = DataLoader(PunktDataset("train", augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(PunktDataset("val"), batch_size=BATCH_SIZE, num_workers=4)

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    # Hintergrund ueberwiegt stark -> positive Pixel hoeher gewichten (Startwert, spaeter Focal Loss)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(100.0, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best = float("inf")
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        val_loss = test(val_dataloader, model, loss_fn)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "unet.pt")
    print("Done!")
