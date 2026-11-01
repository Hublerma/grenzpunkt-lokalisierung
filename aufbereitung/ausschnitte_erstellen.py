"""Schneidet die grossen Pläne (GeoTIFF) in kleine Ausschnitte für das Training von Stufe 1.

Für jeden Plan:
1. Pro Label ein Ausschnitt, in dem der Punkt an einer zufälligen Stelle liegt
   (sonst lernt das Modell "Punkte liegen immer in der Mitte").
2. Zusätzlich zufällige Ausschnitte im Digitalisierungsperimeter (Beispiele mit wenigen oder keinen Punkten).
3. Pläne mit 300 oder 400 DPI werden auf 600 DPI hochgerechnet, damit alle Symbole gleich gross sind.

Ausgabe:
    <AUSGABE>/bilder/<plan>_<nr>.png   Ausschnitt mit GROESSE x GROESSE Pixeln
    <AUSGABE>/punkte/<plan>.csv        eine Zeile pro Punkt und Ausschnitt;
                                       ein Ausschnitt ohne Punkt hat eine Zeile mit leerem x und y

Pixelkoordinaten: Ursprung oben links, x nach rechts, y nach unten, Mitte von Pixel i liegt bei i + 0.5.

Aufruf im Repo-Ordner:
    conda run -n DLgp python aufbereitung/ausschnitte_erstellen.py
"""
import json
import math
import os
import random

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window

DATENSATZ = "data/_aufbereitung/datensatz"
AUSGABE = "data/ausschnitte_256px_600dpi"
GROESSE = 256  # Kantenlänge der Ausschnitte in Pixeln
ZIEL_DPI = 600  # Auflösung aller Ausschnitte
RAND = 16  # Mindestabstand eines Punkts vom Ausschnittrand in Pixeln
ANTEIL_ZUFALL = 0.25  # Anzahl zufälliger Ausschnitte = ANTEIL_ZUFALL * Anzahl Labels
SEED = 42

SPALTEN = ["bild", "plan_id", "split", "typ", "punkt_nr", "a_punkt", "x", "y", "ring_abdeckung"]


def fenster_lesen(geotiff, spalte, zeile, breite):
    """Liest ein quadratisches Fenster (breite x breite Pixel, linke obere Ecke bei spalte/zeile) aus dem GeoTIFF.
    Was ausserhalb des Scans liegt, wird weiss aufgefüllt. Das geschieht von Hand, weil rasterio mit
    boundless=True beim gedrehten GeoTransform der Pläne nicht funktioniert (Erfahrung aus der Aufbereitung)."""
    bild = np.full((breite, breite, 3), 255, dtype=np.uint8)

    # Teil des Fensters, der innerhalb des Scans liegt
    links = max(spalte, 0)
    oben = max(zeile, 0)
    rechts = min(spalte + breite, geotiff.width)
    unten = min(zeile + breite, geotiff.height)

    if rechts > links and unten > oben:
        daten = geotiff.read(window=Window(links, oben, rechts - links, unten - oben))  # Form (Kanäle, Höhe, Breite)
        bild[oben - zeile : unten - zeile, links - spalte : rechts - spalte] = daten.transpose(1, 2, 0)
    return bild


def ausschnitt_schneiden(geotiff, spalte, zeile, faktor):
    """Schneidet einen Ausschnitt aus und rechnet ihn auf GROESSE x GROESSE Pixel bei ZIEL_DPI um."""
    breite_im_scan = GROESSE / faktor  # z. B. 170.67 Scanpixel bei 400 DPI

    # 2 Pixel mehr lesen, damit die bikubische Interpolation auch am rechten und unteren Rand Nachbarpixel hat
    fenster = fenster_lesen(geotiff, spalte, zeile, math.ceil(breite_im_scan) + 2)

    # box = Bereich im Fenster, der auf GROESSE x GROESSE skaliert wird (siehe PIL-Doku zu Image.resize).
    # Damit ist der Faktor exakt 1.0, 1.5 oder 2.0 und passt genau zur Umrechnung der Punkte.
    bild = Image.fromarray(fenster)
    return bild.resize((GROESSE, GROESSE), Image.Resampling.BICUBIC, box=(0, 0, breite_im_scan, breite_im_scan))


def plan_verarbeiten(plan, split, huelle):
    """Erstellt alle Ausschnitte eines Plans und gibt die Tabelle der Punkte zurück."""
    plan_id = plan["plan_id"]
    zufall = random.Random(f"{SEED}_{plan_id}")  # eigener Zufallsgenerator pro Plan, bei jedem Lauf gleich
    faktor = ZIEL_DPI / plan["scan_dpi"]  # 1.0, 1.5 oder 2.0
    breite_im_scan = GROESSE / faktor

    labels = pd.read_csv(f"{DATENSATZ}/labels/{plan_id}.csv", dtype={"punkt_nr": str})
    symbol_score = pd.read_csv(f"{DATENSATZ}/symbol_score/{plan_id}.csv", dtype={"punkt_nr": str})
    labels = labels.merge(symbol_score[["punkt_nr", "ring_abdeckung"]], on="punkt_nr", how="left")

    # Linke obere Ecke jedes Ausschnitts im Scan bestimmen
    ausschnitte = []  # Einträge (typ, spalte, zeile)

    for _, label in labels.iterrows():
        # zufällige Lage des Punkts im Ausschnitt (in Ausschnittpixeln), daraus die Ecke im Scan
        x_im_ausschnitt = zufall.uniform(RAND, GROESSE - RAND)
        y_im_ausschnitt = zufall.uniform(RAND, GROESSE - RAND)
        spalte = math.floor(label["x_px"] - x_im_ausschnitt / faktor)
        zeile = math.floor(label["y_px"] - y_im_ausschnitt / faktor)
        ausschnitte.append(("punkt", spalte, zeile))

    # zufällige Ausschnitte innerhalb des Rechtecks um die Digitalisierungshülle
    x_min, y_min = huelle.min(axis=0)
    x_max, y_max = huelle.max(axis=0)
    for _ in range(round(ANTEIL_ZUFALL * len(labels))):
        spalte = math.floor(zufall.uniform(x_min, x_max - breite_im_scan))
        zeile = math.floor(zufall.uniform(y_min, y_max - breite_im_scan))
        ausschnitte.append(("zufall", spalte, zeile))

    # Ausschnitte schneiden, speichern und die enthaltenen Punkte notieren
    zeilen = []
    with rasterio.open(f"{DATENSATZ}/geotiff/{plan_id}.tif") as geotiff:
        for nr, (typ, spalte, zeile) in enumerate(ausschnitte):
            name = f"{plan_id}_{nr:05d}"
            ausschnitt_schneiden(geotiff, spalte, zeile, faktor).save(f"{AUSGABE}/bilder/{name}.png")

            # alle Labels in Ausschnittkoordinaten umrechnen und die behalten, die im Ausschnitt liegen
            x = (labels["x_px"] - spalte) * faktor
            y = (labels["y_px"] - zeile) * faktor
            im_ausschnitt = (x >= 0) & (x < GROESSE) & (y >= 0) & (y < GROESSE)

            if not im_ausschnitt.any():
                zeilen.append({"bild": name, "plan_id": plan_id, "split": split, "typ": typ})
            for i in labels.index[im_ausschnitt]:
                zeilen.append({
                    "bild": name,
                    "plan_id": plan_id,
                    "split": split,
                    "typ": typ,
                    "punkt_nr": labels.at[i, "punkt_nr"],
                    "a_punkt": labels.at[i, "a_punkt"],
                    "x": round(x[i], 2),
                    "y": round(y[i], 2),
                    "ring_abdeckung": labels.at[i, "ring_abdeckung"],
                })

    # Int64 erlaubt leere Werte, ohne dass aus 0 und 1 Kommazahlen werden
    return pd.DataFrame(zeilen, columns=SPALTEN).astype({"a_punkt": "Int64"})


def main():
    os.makedirs(f"{AUSGABE}/bilder", exist_ok=True)
    os.makedirs(f"{AUSGABE}/punkte", exist_ok=True)

    plaene = pd.read_csv(f"{DATENSATZ}/plaene.csv")
    plaene = plaene[plaene["status"] == "ok"]
    split = pd.read_csv(f"{DATENSATZ}/split.csv").set_index("plan_id")["split"]
    with open(f"{DATENSATZ}/perimeter_px.json") as f:
        perimeter = json.load(f)

    for _, plan in plaene.iterrows():
        plan_id = plan["plan_id"]
        huelle = np.array(perimeter[plan_id]["digitalisierungs_huelle_5m"])
        punkte = plan_verarbeiten(plan, split[plan_id], huelle)
        punkte.to_csv(f"{AUSGABE}/punkte/{plan_id}.csv", index=False)
        print(f"{plan_id}: {punkte['bild'].nunique()} Ausschnitte")


if __name__ == "__main__":
    main()
