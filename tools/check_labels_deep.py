import numpy as np
import os
import glob
from tqdm import tqdm

DATA_DIR = "data/fassade/train"

def check_labels():
    print(f"Prüfe Labels in {DATA_DIR}...")
    files = glob.glob(os.path.join(DATA_DIR, "*", "segment.npy"))
    
    if not files:
        print("Keine Dateien gefunden!")
        return

    max_label_found = -1
    invalid_files = []

    for f in tqdm(files):
        labels = np.load(f)
        
        # Ignoriere -1 (das ist erlaubt für "unlabeled")
        valid_labels = labels[labels != -1]
        
        if len(valid_labels) == 0: continue
        
        current_max = np.max(valid_labels)
        if current_max > max_label_found:
            max_label_found = current_max
            
        if current_max >= 9:
            print(f"\nWARNUNG: Datei {os.path.basename(os.path.dirname(f))} hat Label {current_max}!")
            invalid_files.append(f)

    print("-" * 30)
    print(f"Höchstes gefundenes Label: {max_label_found}")
    
    if max_label_found >= 9:
        print("❌ FEHLER: Labels >= 9 gefunden! Das sprengt das Modell (num_classes=9).")
        print("Bitte führen Sie preprocess_and_split.py erneut aus oder prüfen Sie das Mapping.")
    else:
        print("✅ Alles OK: Alle Labels sind im Bereich 0-8.")

if __name__ == "__main__":
    check_labels()