"""Gemeinsame Funktionen fuer die V2-Notebooks: Ausschnitte mit Kontext direkt aus den GeoTIFFs, Ignore-Maske,
300-DPI-Simulation, Dataset und Inferenz. Unterschiede zu V1 stehen in v2_training.ipynb."""
import json
import math
import os
import random

import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

from grenzpunkte import (BATCH_SIZE, DATENSATZ, GROESSE, RAND, SCHWELLE, ZIEL_DPI, ausschnitt_schneiden, detektieren,
                         device, fenster_lesen, heatmap)
from grenzpunkte import normalize as normalize_v1

KONTEXT = 1024  # px bei 600 DPI, Seitenlaenge des Kontexts um die Mitte des Ausschnitts
RAND_IGNORE = 24  # px, Randstreifen ohne Loss (ein Grenzpunktsymbol hat etwa 15-20 px Radius)
RAND_POSITIV = 32  # px, Mindestabstand eines Labels vom Rand, wenn ein Ausschnitt um ein Label gelegt wird
IGNORE_PN_M = 2.5  # m, Umkreis um nicht digitalisierte PN-Punkte ohne Loss (PN ca. 1 m genau)
LABEL_FREI_PX = 12  # px, um Labels wird immer gerechnet, auch wenn ein nicht digitalisierter PN-Punkt nahe ist
ZYKLUS = 15  # Epochen, bis das Abdeckungsraster einmal vollstaendig durchlaufen ist
ANTEIL_300DPI = 0.5
VAL_ABDECKUNG = 3000  # feste Stichprobe aus dem Abdeckungsraster der val-Plaene
MODELLE = {"v1": ("unet.pt", 3), "v2": ("unet_v2.pt", 6)}  # Gewichte und Anzahl Eingangskanaele
normalize = v2.Normalize(mean=[0.485, 0.456, 0.406] * 2, std=[0.229, 0.224, 0.225] * 2)  # ImageNet, fuer beide Teile

bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(100.0, device=device))  # wie V1, pro Pixel


def loss_fn(pred, y, maske):
    """BCE wie in V1, aber nur ueber die Pixel mit maske = 1 gemittelt."""
    return (bce(pred, y) * maske).sum() / maske.sum().clamp(min=1)


def aufloesung_reduzieren(bild, dpi):
    """Simuliert einen Scan mit tieferer Aufloesung (z. B. 300 DPI): verkleinern mit Flaechenmittel (wie ein Scanner)
    und wieder bikubisch auf 600 DPI bringen. Die Punktkoordinaten bleiben gleich (der Faktor ist exakt)."""
    faktor = ZIEL_DPI / dpi
    klein = bild.resize((round(bild.width / faktor), round(bild.height / faktor)), Image.Resampling.BOX)
    return klein.resize(bild.size, Image.Resampling.BICUBIC)


def kontext_schneiden(geotiff, spalte, zeile, faktor):
    """Kontext zu einem Ausschnitt mit linker oberer Ecke spalte/zeile: KONTEXT x KONTEXT px bei 600 DPI um dieselbe
    Mitte, mit Flaechenmittel auf GROESSE x GROESSE verkleinert. Der Ausschnitt liegt im Kontext in den Pixeln 96-159."""
    rand = round((KONTEXT - GROESSE) / 2 / faktor)  # 384 px bei 600 DPI in Scanpixeln, bei 300/400/600 DPI ganzzahlig
    breite_im_scan = KONTEXT / faktor
    fenster = fenster_lesen(geotiff, spalte - rand, zeile - rand, math.ceil(breite_im_scan) + 2)
    return Image.fromarray(fenster).resize((GROESSE, GROESSE), Image.Resampling.BOX,
                                           box=(0, 0, breite_im_scan, breite_im_scan))


def eingabe(geotiff, spalte, zeile, faktor, dpi=None, mit_kontext=True):
    """Ausschnitt (und Kontext) als Tensor (6 bzw. 3, GROESSE, GROESSE) mit Werten 0..1, noch nicht normiert.
    dpi: den Ausschnitt als Scan mit tieferer Aufloesung simulieren (nur wenn tiefer als die des Scans)."""
    bild = ausschnitt_schneiden(geotiff, spalte, zeile, faktor)
    if dpi is not None and ZIEL_DPI / faktor > dpi:
        bild = aufloesung_reduzieren(bild, dpi)
    bilder = [bild, kontext_schneiden(geotiff, spalte, zeile, faktor)] if mit_kontext else [bild]
    return torch.cat([v2.functional.to_dtype(v2.functional.pil_to_tensor(b), torch.float32, scale=True) for b in bilder])


def plaene_laden(split):
    """Pro Plan des Splits: Faktor, Labels, Digitalisierungshuelle und nicht digitalisierte PN-Punkte (in Scanpixeln)."""
    plaene = pd.read_csv(f"{DATENSATZ}/plaene.csv").query("status == 'ok'").set_index("plan_id")
    splits = pd.read_csv(f"{DATENSATZ}/split.csv").set_index("plan_id")["split"]
    with open(f"{DATENSATZ}/perimeter_px.json") as f:
        perimeter = json.load(f)
    daten = {}
    for plan_id, plan in plaene.iterrows():
        if splits[plan_id] != split:
            continue
        pn_datei = f"{DATENSATZ}/helmert/pn_im_perimeter/{plan_id}.csv"
        mit_pn = os.path.exists(pn_datei)  # fuer Hasle fehlt die PN, dort gibt es keine Maske um PN-Punkte
        pn = pd.read_csv(pn_datei) if mit_pn else pd.DataFrame(columns=["x_px", "y_px", "digitalisiert"])
        daten[plan_id] = {
            "scan_dpi": plan["scan_dpi"],
            "faktor": ZIEL_DPI / plan["scan_dpi"],
            "bodenpixel_m": plan["bodenpixel_m"],
            "labels": pd.read_csv(f"{DATENSATZ}/labels/{plan_id}.csv")[["x_px", "y_px"]].values,
            "huelle": np.array(perimeter[plan_id]["digitalisierungs_huelle_5m"]),
            "mit_pn": mit_pn,
            "pn_offen": pn.loc[pn["digitalisiert"] == 0, ["x_px", "y_px"]].values.astype(float),
        }
    return daten


def maske_erstellen(plan, spalte, zeile):
    """Ignore-Maske eines Ausschnitts: 1 = Loss rechnen, 0 = ignorieren. Ausschnittpixel = (Scanpixel - Ecke) * Faktor."""
    faktor = plan["faktor"]
    maske = Image.new("L", (GROESSE, GROESSE), 0)
    zeichnen = ImageDraw.Draw(maske)
    zeichnen.polygon([tuple(p) for p in (plan["huelle"] - [spalte, zeile]) * faktor], fill=1)  # innerhalb der Huelle
    radius = IGNORE_PN_M / plan["bodenpixel_m"] * faktor  # in Ausschnittpixeln
    for x, y in (plan["pn_offen"] - [spalte, zeile]) * faktor:
        if -radius < x < GROESSE + radius and -radius < y < GROESSE + radius:
            zeichnen.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)
    for x, y in (plan["labels"] - [spalte, zeile]) * faktor:
        if -LABEL_FREI_PX < x < GROESSE + LABEL_FREI_PX and -LABEL_FREI_PX < y < GROESSE + LABEL_FREI_PX:
            zeichnen.ellipse([x - LABEL_FREI_PX, y - LABEL_FREI_PX, x + LABEL_FREI_PX, y + LABEL_FREI_PX], fill=1)
    maske = np.array(maske, dtype=np.float32)
    maske[:RAND_IGNORE], maske[-RAND_IGNORE:], maske[:, :RAND_IGNORE], maske[:, -RAND_IGNORE:] = 0, 0, 0, 0
    return torch.from_numpy(maske)[None]


class AusschnittDataset(Dataset):
    """Ausschnitte (plan_id, spalte, zeile = linke obere Ecke im Scan) direkt aus den GeoTIFFs.
    Die Liste wird vor jeder Epoche mit epoche_vorbereiten neu gezogen (bei val einmal mit feste_auswahl)."""

    def __init__(self, split, augment=False):
        self.plaene = plaene_laden(split)
        self.augment = augment
        self.ausschnitte = []
        self.raster = {}  # Abdeckungsraster pro Zyklus
        self.geotiffs = {}  # wird in jedem DataLoader-Prozess beim ersten Zugriff gefuellt
        self.jitter = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)

    def __getstate__(self):  # offene GeoTIFFs lassen sich nicht an die DataLoader-Prozesse uebergeben
        return {**self.__dict__, "geotiffs": {}}

    def um_punkte(self, punkte, zufall):
        """Ein Ausschnitt pro Punkt (plan_id, x, y), der Punkt liegt an einer zufaelligen Stelle im Ausschnitt."""
        ausschnitte = []
        for plan_id, x, y in punkte:
            faktor = self.plaene[plan_id]["faktor"]
            u, v = zufall.uniform(RAND_POSITIV, GROESSE - RAND_POSITIV), zufall.uniform(RAND_POSITIV, GROESSE - RAND_POSITIV)
            ausschnitte.append((plan_id, math.floor(x - u / faktor), math.floor(y - v / faktor)))
        return ausschnitte

    def abdeckung(self, zufall):
        """Raster aus lueckenlos aneinanderliegenden Ausschnitten ueber die Huelle jedes Plans, pro Plan mit
        zufaelligem Versatz (damit die Grenzen der Ausschnitte nicht in jedem Zyklus gleich liegen)."""
        ausschnitte = []
        for plan_id, plan in self.plaene.items():
            seite = GROESSE / plan["faktor"]  # Seitenlaenge in Scanpixeln
            huelle = Path(plan["huelle"])
            (x_min, y_min), (x_max, y_max) = plan["huelle"].min(axis=0), plan["huelle"].max(axis=0)
            for y in np.arange(y_min - zufall.uniform(0, seite), y_max, seite):
                for x in np.arange(x_min - zufall.uniform(0, seite), x_max, seite):
                    if huelle.intersects_bbox(Bbox.from_extents(x, y, x + seite, y + seite)):
                        ausschnitte.append((plan_id, math.floor(x), math.floor(y)))
        zufall.shuffle(ausschnitte)
        return ausschnitte

    def epoche_vorbereiten(self, epoche):
        """Zieht die Ausschnitte einer Epoche: alle Labels und ein Teil des Abdeckungsrasters."""
        zufall = random.Random(epoche)
        zyklus = epoche // ZYKLUS
        if zyklus not in self.raster:
            self.raster = {zyklus: self.abdeckung(random.Random(f"zyklus_{zyklus}"))}
        raster = self.raster[zyklus][epoche % ZYKLUS :: ZYKLUS]  # jeder Teil kommt pro Zyklus genau einmal dran
        labels = [(p, x, y) for p, plan in self.plaene.items() for x, y in plan["labels"]]
        self.ausschnitte = self.um_punkte(labels, zufall) + raster
        return len(labels), len(raster)

    def feste_auswahl(self):
        """Feste Auswahl fuer die Validierung: alle Labels und VAL_ABDECKUNG Raster-Ausschnitte (immer gleich)."""
        zufall = random.Random(0)
        labels = [(p, x, y) for p, plan in self.plaene.items() for x, y in plan["labels"]]
        self.ausschnitte = self.um_punkte(labels, zufall) + self.abdeckung(zufall)[:VAL_ABDECKUNG]

    def __len__(self):
        return len(self.ausschnitte)

    def __getitem__(self, idx):
        plan_id, spalte, zeile = self.ausschnitte[idx]
        plan = self.plaene[plan_id]
        faktor = plan["faktor"]
        if plan_id not in self.geotiffs:
            self.geotiffs[plan_id] = rasterio.open(f"{DATENSATZ}/geotiff/{plan_id}.tif")
        dpi = 300 if self.augment and random.random() < ANTEIL_300DPI else None
        image = eingabe(self.geotiffs[plan_id], spalte, zeile, faktor, dpi)
        punkte = (plan["labels"] - [spalte, zeile]) * faktor
        label = heatmap(punkte[((punkte >= 0) & (punkte < GROESSE)).all(axis=1)], GROESSE, GROESSE)
        maske = maske_erstellen(plan, spalte, zeile)
        if self.augment:
            # Farbe: Ausschnitt und Kontext nebeneinander, damit beide gleich veraendert werden
            nebeneinander = self.jitter(torch.cat([image[:3], image[3:]], dim=2))
            image = torch.cat([nebeneinander[:, :, :GROESSE], nebeneinander[:, :, GROESSE:]])
            # Drehen/Spiegeln um die Mitte: Ausschnitt und Kontext haben dieselbe Mitte und bleiben deckungsgleich
            k = random.randint(0, 3)
            image, label, maske = (torch.rot90(t, k, (1, 2)) for t in (image, label, maske))
            if random.random() < 0.5:
                image, label, maske = (t.flip(2) for t in (image, label, maske))
        return normalize(image), label, maske


def modell_laden(version="v2"):
    datei, kanaele = MODELLE[version]
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=kanaele, classes=1)
    model.load_state_dict(torch.load(datei, map_location=device))
    return model.to(device).eval()


def plan_detektieren(model, plan_id, scan_dpi, schwelle=SCHWELLE, dpi=None):
    """Wie grenzpunkte.plan_detektieren, aber mit Kontext (6 Kanaele) und optional simulierter tieferer Aufloesung.
    Rechnet auch das V1-Modell (3 Kanaele), damit sich V1 und V2 auf 300 DPI vergleichen lassen."""
    mit_kontext = model.encoder.conv1.in_channels == 6
    norm = normalize if mit_kontext else normalize_v1
    faktor = ZIEL_DPI / scan_dpi
    kern = round((GROESSE - 2 * RAND) / faktor)
    rand = round(RAND / faktor)
    detektionen = []
    with rasterio.open(f"{DATENSATZ}/geotiff/{plan_id}.tif") as geotiff, torch.no_grad():
        kacheln = [(s, z) for z in range(0, geotiff.height, kern) for s in range(0, geotiff.width, kern)]
        print(f"{plan_id}: {len(kacheln)} Kacheln")
        for k in range(0, len(kacheln), BATCH_SIZE):
            batch = kacheln[k : k + BATCH_SIZE]
            X = torch.stack([norm(eingabe(geotiff, s - rand, z - rand, faktor, dpi, mit_kontext)) for s, z in batch])
            heatmaps = torch.sigmoid(model(X.to(device)))[:, 0].cpu().numpy()
            for (s, z), hm in zip(batch, heatmaps):
                for x, y, wert in detektieren(hm, schwelle):
                    x_scan, y_scan = s - rand + x / faktor, z - rand + y / faktor
                    if s <= x_scan < s + kern and z <= y_scan < z + kern:
                        detektionen.append((x_scan, y_scan, wert))
    return np.array(detektionen).reshape(-1, 3)
