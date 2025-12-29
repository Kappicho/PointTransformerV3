import numpy as np
import os
import glob
import random
import shutil
from tqdm import tqdm

# --- EINSTELLUNGEN ---
DATA_ROOT = "data/fassade"
RAW_DIR = os.path.join(DATA_ROOT, "raw")
CACHE_DIR = os.path.join(DATA_ROOT, "cache")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
META_DIR = os.path.join(DATA_ROOT, "meta")
VAL_RATIO = 0.2 
MAPPING = {i: i for i in range(9)}

# Loop-Faktor für Training (Vervielfachung)
LOOP_FACTOR = 10

def process_and_split():
    # 1. Aufräumen der Zielordner
    if os.path.exists(TRAIN_DIR): shutil.rmtree(TRAIN_DIR)
    if os.path.exists(VAL_DIR): shutil.rmtree(VAL_DIR)
    
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 2. Raw Dateien finden
    raw_files = glob.glob(os.path.join(RAW_DIR, "*.txt"))
    if not raw_files:
        print(f"FEHLER: Keine .txt Dateien in {RAW_DIR} gefunden!")
        return
    
    all_names = [os.path.splitext(os.path.basename(f))[0] for f in raw_files]
    name_to_path = {os.path.splitext(os.path.basename(f))[0]: f for f in raw_files}

    # 3. SPLIT BESTIMMEN
    train_names = []
    val_names = []
    
    train_list_path = os.path.join(META_DIR, "train.txt")
    val_list_path = os.path.join(META_DIR, "val.txt")
    
    if os.path.exists(train_list_path) and os.path.exists(val_list_path):
        print("=> Nutze existierende Split-Listen aus 'meta/'")
        with open(train_list_path, 'r') as f:
            train_names = [l.strip() for l in f.readlines() if l.strip()]
        with open(val_list_path, 'r') as f:
            val_names = [l.strip() for l in f.readlines() if l.strip()]
    else:
        print("=> Keine Listen gefunden. Erstelle zufälligen Split...")
        random.seed(42)
        random.shuffle(all_names)
        if len(all_names) == 1:
            train_names = all_names
            val_names = all_names
        else:
            val_size = max(1, int(len(all_names) * VAL_RATIO))
            val_names = all_names[:val_size]
            train_names = all_names[val_size:]
        
        with open(train_list_path, "w") as f: f.write("\n".join(train_names))
        with open(val_list_path, "w") as f: f.write("\n".join(val_names))

    print(f"=== TRAINING: {len(train_names)} Scans | VALIDIERUNG: {len(val_names)} Scans ===")

    # 4. VERARBEITUNG MIT SYMLINKS
    def prepare_scene(name, target_root_dir, num_copies):
        fpath = name_to_path.get(name)
        if not fpath: return

        # Cache erstellen (falls noch nicht da)
        scene_cache_dir = os.path.join(CACHE_DIR, name)
        if not os.path.exists(os.path.join(scene_cache_dir, "coord.npy")):
            try:
                data = np.loadtxt(fpath)
                xyz = data[:, 0:3].astype(np.float32)
                xyz -= np.mean(xyz, axis=0)
                
                intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
                if data.shape[1] > 3:
                    inten = data[:, 3].astype(np.float32)
                    if np.max(inten) > 1.0: inten /= np.max(inten)
                    intensity = inten[:, None]

                final_labels = np.zeros((xyz.shape[0]), dtype=np.int64)
                if data.shape[1] > 4:
                    raw = data[:, 4].astype(np.int64)
                    final_labels = np.full(raw.shape, -1, dtype=np.int64)
                    for r, t in MAPPING.items(): final_labels[raw == r] = t

                rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
                if data.shape[1] > 7:
                    c = data[:, 5:8].astype(np.float32)
                    if np.max(c) <= 1.1: c *= 255.0
                    rgb = c.astype(np.uint8)

                normal = np.zeros_like(xyz)
                if data.shape[1] >= 11:
                    normal = data[:, 8:11].astype(np.float32)

                os.makedirs(scene_cache_dir, exist_ok=True)
                np.save(os.path.join(scene_cache_dir, "coord.npy"), xyz)
                np.save(os.path.join(scene_cache_dir, "color.npy"), rgb)
                np.save(os.path.join(scene_cache_dir, "normal.npy"), normal)
                np.save(os.path.join(scene_cache_dir, "strength.npy"), intensity)
                np.save(os.path.join(scene_cache_dir, "segment.npy"), final_labels)
            except Exception as e:
                print(f"Fehler bei {name}: {e}")
                return

        # SYMLINKS ERSTELLEN (Statt Kopieren!)
        # Wir verlinken jede Datei einzeln, um Speicher zu sparen.
        # Absolute Pfade sind sicherer für Symlinks.
        abs_cache_dir = os.path.abspath(scene_cache_dir)
        
        for i in range(num_copies):
            link_name = f"{name}_v{i:02d}" if num_copies > 1 else name
            target_dir = os.path.join(target_root_dir, link_name)
            
            if os.path.exists(target_dir): shutil.rmtree(target_dir)
            os.makedirs(target_dir, exist_ok=True)
            
            # Alle .npy Dateien verlinken
            for npy_file in glob.glob(os.path.join(abs_cache_dir, "*.npy")):
                base_name = os.path.basename(npy_file)
                link_path = os.path.join(target_dir, base_name)
                try:
                    os.symlink(npy_file, link_path)
                except FileExistsError:
                    pass

    # Train verarbeiten
    print("-> Linke Training Daten (Symlinks)...")
    for name in tqdm(train_names):
        prepare_scene(name, TRAIN_DIR, LOOP_FACTOR)
        
    # Val verarbeiten
    print("-> Linke Validierungs Daten (Symlinks)...")
    for name in tqdm(val_names):
        prepare_scene(name, VAL_DIR, 1)

    num_links = len(glob.glob(os.path.join(TRAIN_DIR, "*")))
    print(f"✅ Fertig. Train Ordner enthält {num_links} virtuelle Szenen (0 Speicherverbrauch).")

if __name__ == "__main__":
    process_and_split()
