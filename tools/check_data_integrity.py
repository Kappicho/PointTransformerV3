import numpy as np
import os
import glob
from tqdm import tqdm

# Pfad zu den Trainingsdaten
DATA_DIR = "data/fassade/train"

def check_integrity():
    print(f"🔍 Starte Tiefenprüfung in {DATA_DIR} ...")
    # Wir suchen in allen Unterordnern (den _v00 etc. Ordnern)
    # Wir nehmen nur die Originale (_v00), da die anderen nur Symlinks sind
    scene_dirs = glob.glob(os.path.join(DATA_DIR, "*_v00"))
    
    if not scene_dirs:
        # Fallback falls keine Versionierung
        scene_dirs = glob.glob(os.path.join(DATA_DIR, "*"))
        # Filtere Dateien raus, nimm nur Ordner
        scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]

    print(f"Prüfe {len(scene_dirs)} Basis-Szenen...")
    
    error_count = 0

    for scene_dir in tqdm(scene_dirs):
        name = os.path.basename(scene_dir)
        
        # 1. Pfade
        f_coord = os.path.join(scene_dir, "coord.npy")
        f_feat = os.path.join(scene_dir, "color.npy") # und strength etc.
        f_strength = os.path.join(scene_dir, "strength.npy")
        f_label = os.path.join(scene_dir, "segment.npy")
        
        try:
            # 2. Check: Existenz
            if not os.path.exists(f_coord):
                print(f"⚠️ {name}: coord.npy fehlt!")
                continue
                
            # 3. Check: Koordinaten (NaN, Inf, Outlier)
            coord = np.load(f_coord)
            if np.isnan(coord).any():
                print(f"❌ ALARM {name}: Koordinaten enthalten NaN!")
                error_count += 1
            if np.isinf(coord).any():
                print(f"❌ ALARM {name}: Koordinaten enthalten Infinite!")
                error_count += 1
            
            # Check auf extreme Werte (größer als 1km = wahrscheinlich Fehler)
            max_val = np.max(np.abs(coord))
            if max_val > 1000:
                print(f"⚠️ WARNUNG {name}: Extreme Koordinaten gefunden (Max: {max_val:.1f}m). Zentrierung fehlgeschlagen?")
            
            # 4. Check: Features (Farbe, Stärke)
            if os.path.exists(f_strength):
                strength = np.load(f_strength)
                if np.isnan(strength).any() or np.isinf(strength).any():
                    print(f"❌ ALARM {name}: Intensität enthält NaN/Inf!")
                    error_count += 1
                    
            if os.path.exists(f_feat):
                color = np.load(f_feat)
                if np.isnan(color).any() or np.isinf(color).any():
                    print(f"❌ ALARM {name}: Farbe enthält NaN/Inf!")
                    error_count += 1

            # 5. Check: Labels (Bereich 0-8)
            if os.path.exists(f_label):
                label = np.load(f_label)
                # Ignoriere -1
                valid_l = label[label != -1]
                if len(valid_l) > 0:
                    if np.max(valid_l) > 8 or np.min(valid_l) < 0:
                        print(f"❌ ALARM {name}: Ungültige Labels gefunden! (Min: {np.min(valid_l)}, Max: {np.max(valid_l)})")
                        error_count += 1

        except Exception as e:
            print(f"❌ CRASH {name}: Konnte Datei nicht lesen: {e}")
            error_count += 1

    print("-" * 30)
    if error_count == 0:
        print("✅ Alles sauber. Keine korrupten Daten gefunden.")
    else:
        print(f"⛔ {error_count} Fehler gefunden! Bitte beheben.")

if __name__ == "__main__":
    check_integrity()
