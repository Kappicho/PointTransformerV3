import numpy as np
import os
import glob
from tqdm import tqdm

ROOT = "data/fassade"

def check(folder):
    print(f"🔎 Prüfe Ordner: {folder} ...")
    files = glob.glob(os.path.join(folder, "**", "segment.npy"), recursive=True)
    
    if not files:
        print("Keine .npy Dateien gefunden.")
        return

    for f in tqdm(files):
        try:
            labels = np.load(f)
            valid = labels[labels != -1]
            
            if len(valid) == 0: continue
            
            max_val = valid.max()
            min_val = valid.min()
            
            if max_val >= 9:
                print(f"\n🚨 ALARM in {f}: Max Label ist {max_val} (Erlaubt: 0-8)")
            if min_val < 0: # Außer -1
                print(f"\n🚨 ALARM in {f}: Min Label ist {min_val}")
                
        except Exception as e:
            print(f"Fehler beim Lesen von {f}: {e}")

print("=== CHECK TRAIN ===")
check(os.path.join(ROOT, "train"))

print("\n=== CHECK VAL ===")
check(os.path.join(ROOT, "val"))