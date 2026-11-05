"""Gemeinsame Funktionen fuer die Notebooks (V1).
Hier steht nur, was mehrere Notebooks brauchen oder was importierbar sein muss: Die DataLoader-Worker laufen unter
macOS als eigene Prozesse und finden ein Dataset nur, wenn es in einer .py-Datei steht (nicht in einer Notebook-Zelle).
Pixelkoordinaten: Ursprung oben links, x nach rechts, y nach unten, Mitte von Pixel i liegt bei i + 0.5."""
import glob
import math
import random

import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from rasterio.windows import Window
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

DATENSATZ = "data/_aufbereitung/datensatz"
DATA = "data/ausschnitte_256px_600dpi"  # erstellt mit 01_ausschnitte_erstellen.ipynb
GROESSE = 256  # Kantenlaenge der Ausschnitte in Pixeln
ZIEL_DPI = 600  # Aufloesung aller Ausschnitte
SIGMA = 2.0  # px, Breite der Gaussglocke
BATCH_SIZE = 16
MIN_ABSTAND = 5  # px, Mindestabstand zwischen zwei Detektionen
MAX_ABSTAND = 8  # px, Detektion und Label bilden nur bis zu diesem Abstand ein Paar
SCHWELLE = 0.9  # Standard-Mindestwert der Heatmap fuer eine Detektion
RAND = 32  # px, bei der Inferenz wird von jeder Kachel nur der innere Teil verwendet
device = torch.device("mps")
normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet


# ---------------------------------------------------------------- Ausschnitte aus den GeoTIFFs schneiden
def fenster_lesen(geotiff, spalte, zeile, breite):
    """Liest ein quadratisches Fenster (breite x breite Pixel, linke obere Ecke bei spalte/zeile) aus dem GeoTIFF.
    Was ausserhalb des Scans liegt, wird weiss aufgefuellt. Das geschieht von Hand, weil rasterio mit
    boundless=True beim gedrehten GeoTransform der Plaene nicht funktioniert."""
    bild = np.full((breite, breite, 3), 255, dtype=np.uint8)
    links, oben = max(spalte, 0), max(zeile, 0)
    rechts, unten = min(spalte + breite, geotiff.width), min(zeile + breite, geotiff.height)
    if rechts > links and unten > oben:
        daten = geotiff.read(window=Window(links, oben, rechts - links, unten - oben))  # (Kanaele, Hoehe, Breite)
        bild[oben - zeile : unten - zeile, links - spalte : rechts - spalte] = daten.transpose(1, 2, 0)
    return bild


def ausschnitt_schneiden(geotiff, spalte, zeile, faktor):
    """Schneidet einen Ausschnitt aus und rechnet ihn auf GROESSE x GROESSE Pixel bei ZIEL_DPI um."""
    breite_im_scan = GROESSE / faktor  # z. B. 170.67 Scanpixel bei 400 DPI
    # 2 Pixel mehr lesen, damit die bikubische Interpolation auch am rechten und unteren Rand Nachbarpixel hat
    fenster = fenster_lesen(geotiff, spalte, zeile, math.ceil(breite_im_scan) + 2)
    # box = Bereich im Fenster, der skaliert wird (PIL-Doku zu Image.resize): Faktor exakt 1.0, 1.5 oder 2.0
    return Image.fromarray(fenster).resize((GROESSE, GROESSE), Image.Resampling.BICUBIC,
                                           box=(0, 0, breite_im_scan, breite_im_scan))


# ---------------------------------------------------------------- Training
def heatmap(punkte, h, w):
    """Gaussglocke pro Punkt (Mitte von Pixel i liegt bei i + 0.5), ueberlappende Glocken per Maximum."""
    x = torch.arange(w) + 0.5
    y = torch.arange(h) + 0.5
    hm = torch.zeros(1, h, w)
    for px, py in punkte:
        hm = torch.maximum(hm, torch.exp(-((x[None, :] - px) ** 2 + (y[:, None] - py) ** 2) / (2 * SIGMA**2)))
    return hm


class PunktDataset(Dataset):
    """Ausschnitte eines Splits mit Ziel-Heatmap (Aufbau nach dem PyTorch-Tutorial Datasets & DataLoaders)."""

    def __init__(self, split, augment=False):
        df = pd.concat(pd.read_csv(f) for f in glob.glob(f"{DATA}/punkte/*.csv"))
        df = df[df["split"] == split]
        self.bilder = df["bild"].unique()  # vor dropna, damit Ausschnitte ohne Punkt bleiben
        df = df.dropna(subset=["x"])
        self.punkte = {b: g[["x", "y"]].values for b, g in df.groupby("bild")}
        self.augment = augment
        self.jitter = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)

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
        return normalize(image), label


# ---------------------------------------------------------------- Detektion in der Heatmap
def versatz(links, mitte, rechts):
    """Lage des Scheitels einer Parabel durch die Logarithmen von drei Nachbarwerten, relativ zum mittleren Pixel.
    Fuer eine Gaussglocke exakt, denn der Logarithmus einer Gaussglocke ist eine Parabel."""
    links, mitte, rechts = np.log(np.maximum([links, mitte, rechts], 1e-10))
    nenner = links - 2 * mitte + rechts
    return 0.5 * (links - rechts) / nenner if nenner < 0 else 0.0


def subpixel(heatmap, zeile, spalte):
    """Verfeinert ein Maximum auf Subpixel, getrennt in x und y. Gibt x, y zurueck."""
    hoehe, breite = heatmap.shape
    dx = versatz(*heatmap[zeile, spalte - 1 : spalte + 2]) if 0 < spalte < breite - 1 else 0.0
    dy = versatz(*heatmap[zeile - 1 : zeile + 2, spalte]) if 0 < zeile < hoehe - 1 else 0.0
    return spalte + 0.5 + dx, zeile + 0.5 + dy


def detektieren(heatmap, schwelle):
    """Findet die Punkte in einer Heatmap. Gibt ein Array (n, 3) mit x, y und Heatmap-Wert zurueck."""
    maxima = peak_local_max(heatmap, min_distance=MIN_ABSTAND, threshold_abs=schwelle, exclude_border=False)
    return np.array([(*subpixel(heatmap, z, s), heatmap[z, s]) for z, s in maxima]).reshape(-1, 3)


def zuordnen(detektionen, labels):
    """Ordnet Detektionen und Labels eins zu eins zu (kleinste Summe der Abstaende).
    Gibt die Paare (Index Detektion, Index Label, Abstand) zurueck, die hoechstens MAX_ABSTAND auseinander liegen."""
    if len(detektionen) == 0 or len(labels) == 0:
        return []
    abstand = np.linalg.norm(detektionen[:, None, :] - labels[None, :, :], axis=2)
    kosten = np.where(abstand <= MAX_ABSTAND, abstand, 1e6)  # zu weit entfernte Paare praktisch verbieten
    d, l = linear_sum_assignment(kosten)
    return [(i, j, abstand[i, j]) for i, j in zip(d, l) if abstand[i, j] <= MAX_ABSTAND]


# ---------------------------------------------------------------- Inferenz auf einem ganzen Plan
def modell_laden(datei="unet.pt"):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(datei, map_location=device))
    return model.to(device).eval()


def plan_detektieren(model, plan_id, scan_dpi, schwelle=SCHWELLE):
    """Sagt einen ganzen Plan in ueberlappenden Kacheln vorher (gleiche Schneide-Funktion wie beim Training).
    Gibt die Detektionen in Scanpixeln mit ihrem Heatmap-Wert zurueck (Array n x 3: x, y, Wert)."""
    faktor = ZIEL_DPI / scan_dpi
    kern = round((GROESSE - 2 * RAND) / faktor)  # innerer Teil einer Kachel in Scanpixeln
    rand = round(RAND / faktor)  # Rand einer Kachel in Scanpixeln
    detektionen = []
    with rasterio.open(f"{DATENSATZ}/geotiff/{plan_id}.tif") as geotiff, torch.no_grad():
        # die inneren Teile (kern x kern) der Kacheln decken den Plan lueckenlos ab
        kacheln = [(s, z) for z in range(0, geotiff.height, kern) for s in range(0, geotiff.width, kern)]
        print(f"{plan_id}: {len(kacheln)} Kacheln")
        for k in range(0, len(kacheln), BATCH_SIZE):
            batch = kacheln[k : k + BATCH_SIZE]
            bilder = [ausschnitt_schneiden(geotiff, s - rand, z - rand, faktor) for s, z in batch]
            X = torch.stack([normalize(v2.functional.to_dtype(v2.functional.pil_to_tensor(b), torch.float32, scale=True))
                             for b in bilder])
            heatmaps = torch.sigmoid(model(X.to(device)))[:, 0].cpu().numpy()
            for (s, z), hm in zip(batch, heatmaps):
                for x, y, wert in detektieren(hm, schwelle):
                    x_scan, y_scan = s - rand + x / faktor, z - rand + y / faktor  # Kachelpixel -> Scanpixel
                    if s <= x_scan < s + kern and z <= y_scan < z + kern:  # nur Detektionen im inneren Teil
                        detektionen.append((x_scan, y_scan, wert))
    return np.array(detektionen).reshape(-1, 3)
