import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse

# --- KLASSEN DEFINITION (Muss zur Config passen) ---
CLASSES = [
    "Sonstiges",    # 0
    "Putz",         # 1
    "Beton",        # 2
    "Backstein",    # 3
    "Dachziegel",   # 4
    "Stein"         # 5
]

def generate_matrix(exp_dir):
    export_folder = os.path.join(exp_dir, "cloudcompare_export")
    if not os.path.exists(export_folder):
        print(f"FEHLER: Ordner {export_folder} existiert nicht.")
        print("Bitte führen Sie zuerst das Export-Skript (report_csv/export) aus!")
        return

    # Suche alle prediction files
    files = glob.glob(os.path.join(export_folder, "*_prediction.txt"))
    if not files:
        print("Keine *_prediction.txt Dateien gefunden.")
        return

    print(f"Lese {len(files)} Dateien für die Matrix...")

    all_gt = []
    all_pred = []
    total_points = 0

    for f_path in files:
        try:
            # Lese Datei. Format war: X Y Z R G B GroundTruth Prediction IsError
            # Wir brauchen Spalte 6 (GT) und 7 (Pred) -> Indizes sind 0-basiert
            # sep=' ' für Leerzeichen-getrennte Dateien
            data = pd.read_csv(f_path, sep=' ', skiprows=1, header=None, usecols=[6, 7], names=['GT', 'Pred'])
            
            all_gt.append(data['GT'].values)
            all_pred.append(data['Pred'].values)
            total_points += len(data)
        except Exception as e:
            print(f"Warnung bei {os.path.basename(f_path)}: {e}")

    if total_points == 0:
        print("Keine Datenpunkte gefunden.")
        return

    # Zusammenfügen aller Arrays
    y_true = np.concatenate(all_gt)
    y_pred = np.concatenate(all_pred)

    print(f"Berechne Confusion Matrix basierend auf {total_points} Punkten...")
    
    # Berechne Matrix (labels=range stellt sicher, dass alle Klassen 0-5 auftauchen, auch wenn leer)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    
    # Normalisierung (Recall / Zeilenweise): Wieviel % der wahren Klasse wurden wie klassifiziert?
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # NaN durch 0 ersetzen (falls eine Klasse in Ground Truth gar nicht vorkommt)
    cm_norm = np.nan_to_num(cm_norm)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 8))
    
    # Heatmap erstellen
    # fmt='.1%' zeigt eine Nachkommastelle (z.B. 95.4%)
    ax = sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                     xticklabels=CLASSES, yticklabels=CLASSES,
                     cbar_kws={'label': 'Anteil der Klassifikation (Recall)'},
                     vmin=0, vmax=1) # Skala fest von 0 bis 100%

    plt.ylabel('Wahre Klasse (Ground Truth)', fontsize=12)
    plt.xlabel('Vorhergesagte Klasse (Prediction)', fontsize=12)
    
    run_name = os.path.basename(exp_dir.rstrip('/'))
    plt.title(f'Confusion Matrix: {run_name}', fontsize=14)
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 1. Run-Namen aus dem Pfad holen (z.B. "run21")
    run_name = os.path.basename(exp_dir.rstrip('/'))

    # 2. Speichern mit dynamischem Namen
    save_path = os.path.join(exp_dir, f"confusion_matrix_{run_name}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Matrix als Bild gespeichert: {save_path}")
    # plt.show() # Optional, falls GUI vorhanden

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", type=str, required=True, help="Pfad zum Experiment-Ordner")
    args = parser.parse_args()
    
    generate_matrix(args.exp_dir)