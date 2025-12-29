import torch
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --- IMPORT FIX: Damit Python 'pointcept' im Hauptordner findet ---
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets import build_dataset

# --- KONFIGURATION ---
# Wie viele Punkte sollen PRO KLASSE gesammelt werden?
POINTS_PER_CLASS = 500  

# Deine Klassen (Namen müssen zur Reihenfolge der IDs 0, 1, 2... passen)
CLASSES = [
    "Sonstiges", "Putz", "Beton", "Backstein", "Dachziegel", "Stein"
]

COLORS = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#8c564b', '#9467bd']

# --- HOOK FÜR LATENT SPACE ---
activation = {}
# --- HOOK FÜR LATENT SPACE (Verbessert für Sparse Tensor) ---
activation = {}
def get_activation(name):
    def hook(model, input, output):
        # input ist ein Tuple, das Datenobjekt ist an Stelle 0
        x = input[0]
        
        # Prüfung: Ist es ein spezieller SparseConvTensor?
        if hasattr(x, "features"):
            # Ja -> Wir wollen nur die Features (Zahlenwerte)
            activation[name] = x.features.detach()
        else:
            # Nein -> Es ist wohl ein normaler Tensor
            activation[name] = x.detach()
    return hook

# --- COLLATE FUNKTION (Datenvorbereitung) ---
def collate_fn(batch):
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x
    
    data_dict = {}
    coords_list = [to_tensor(b["coord"]) for b in batch]
    coords_flat = torch.cat(coords_list, dim=0)
    
    # Grid & Offset Handling
    offset_list = [c.shape[0] for c in coords_list]
    offset = torch.cumsum(torch.tensor(offset_list), dim=0).int()
    grid_list = [to_tensor(b["grid_coord"]) for b in batch]
    grid_flat = torch.cat(grid_list, dim=0)
    
    batch_list = []
    for i, g in enumerate(grid_list):
        batch_list.append(torch.full((g.shape[0],), i, dtype=torch.long))
    batch_flat = torch.cat(batch_list, dim=0)
    
    # Feature Handling (Flexibel für verschiedene Inputs)
    if "feat" in batch[0]:
        feat_flat = torch.cat([to_tensor(b["feat"]) for b in batch], dim=0).float()
    else:
        feat_flat_list = []
        for b in batch:
            f = []
            # Wir nehmen alles was da ist: Farbe, Normale, Intensität
            if "color" in b: f.append(to_tensor(b["color"]))
            if "normal" in b: f.append(to_tensor(b["normal"]))
            if "strength" in b: f.append(to_tensor(b["strength"]))
            
            if len(f) > 0:
                feat_flat_list.append(torch.cat(f, dim=1))
            else:
                # Fallback: Leere Features
                feat_flat_list.append(torch.zeros((b["coord"].shape[0], 1))) 
        feat_flat = torch.cat(feat_flat_list, dim=0).float()
        
    segment_flat = torch.cat([to_tensor(b["segment"]) for b in batch], dim=0).long()
    
    return {
        "coord": coords_flat, 
        "grid_coord": grid_flat, 
        "batch": batch_flat,
        "segment": segment_flat, 
        "feat": feat_flat, 
        "offset": offset
    }

def visualize_latent_space(exp_dir, mode='latent'):
    config_path = os.path.join(exp_dir, "config.py")
    best_pth = os.path.join(exp_dir, "model/model_best.pth")
    last_pth = os.path.join(exp_dir, "model/model_last.pth")
    
    if os.path.exists(best_pth):
        checkpoint_path = best_pth
    elif os.path.exists(last_pth):
        checkpoint_path = last_pth
    else:
        print(f"❌ Kein Checkpoint in {exp_dir} gefunden.")
        return

    print(f"=> Lade Config: {config_path}")
    cfg = Config.fromfile(config_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("=> Baue Modell...")
    model = build_model(cfg.model).to(device)

   # ... nach: model = build_model(cfg.model).to(device)

    print("\n" + "="*40)
    print("🔍 TIEFEN-INSPEKTION DES BACKBONES:")
    if hasattr(model, 'backbone'):
        for name, module in model.backbone.named_children():
            print(f"  Layer: '{name}' -> Typ: {type(module).__name__}")
    else:
        print("  (Kein Backbone-Attribut gefunden?)")
    print("="*40 + "\n")
    
    # sys.exit() # Entfernen Sie das # am Anfang, wenn Sie NUR gucken wollen
    
    print(f"=> Lade Weights von {os.path.basename(checkpoint_path)}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Hook registrieren, falls wir in den Latent Space wollen
    if mode == 'latent':
        # Fall A: Standard PointTransformer (seg_head)
        if hasattr(model, 'seg_head'):
            model.seg_head.register_forward_hook(get_activation('latent_features'))
            print("=> Hook aktiv: Greife Features vor 'seg_head' ab.")
        
        # Fall B: Dein Modell (SpUNet / MinkUNet mit 'final')
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'final'):
            model.backbone.final.register_forward_hook(get_activation('latent_features'))
            print("=> Hook aktiv: Greife Features vor 'backbone.final' ab.")
            
        else:
            print("⚠️ Warnung: Konnte keinen passenden Layer finden. Wechsle zu 'logits'.")
            mode = 'logits'

    print("=> Lade Validierungs-Datensatz...")
    dataset = build_dataset(cfg.data.val)
    
    # Speicher für unsere Punkte pro Klasse
    class_features = {i: [] for i in range(len(CLASSES))}
    class_counts = {i: 0 for i in range(len(CLASSES))}
    
    print(f"=> Sammle Daten (Ziel: {POINTS_PER_CLASS} Punkte pro Klasse)...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            # Abbruch, wenn wir für alle Klassen genug haben
            if all(count >= POINTS_PER_CLASS for count in class_counts.values()): 
                print("Alle Klassen voll. Stoppe Datensammlung.")
                break
            
            try:
                data_raw = dataset[i]
                data = collate_fn([data_raw])
            except Exception as e:
                print(f"⚠️ Überspringe Szene {i} (Fehler beim Laden): {e}")
                continue

            # Auf GPU schieben
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
            
            # Forward Pass
            output = model(data)
            
            # Daten extrahieren je nach Modus
            if mode == 'latent' and 'latent_features' in activation:
                feats = activation['latent_features'].cpu().numpy()
            elif mode == 'input':
                feats = data['feat'].cpu().numpy()
            else: # logits
                feats = torch.softmax(output['seg_logits'], dim=1).cpu().numpy()
            
            labels = data['segment'].cpu().numpy()

            # DEBUG: Zeige welche IDs in dieser Szene vorkommen
            unique_ids = np.unique(labels)
            # Nur printen, wenn wir noch nicht voll sind, um Spam zu vermeiden
            if i < 3: 
                print(f" Szene {i}: Vorhandene IDs = {unique_ids}")

            # Punkte in die Buckets sortieren
            for class_id in range(len(CLASSES)):
                # Wenn Bucket voll, überspringen
                if class_counts[class_id] >= POINTS_PER_CLASS: continue
                
                # Maske für die aktuelle Klasse
                mask = (labels == class_id)
                
                if np.any(mask):
                    f_cls = feats[mask] # Features dieser Klasse
                    
                    # Berechnen wie viele wir noch brauchen
                    needed = POINTS_PER_CLASS - class_counts[class_id]
                    take = min(len(f_cls), needed)
                    
                    # Zufällige Auswahl (Sampling)
                    idx = np.random.choice(len(f_cls), size=take, replace=False)
                    
                    class_features[class_id].append(f_cls[idx])
                    class_counts[class_id] += take
            
            # Kurzer Status-Print
            status_str = ", ".join([f"{CLASSES[k]}:{class_counts[k]}" for k in range(len(CLASSES))])
            sys.stdout.write(f"\r Szene {i} verarbeitet. Status: [{status_str}]")
            sys.stdout.flush()

    print("\n=> Datensammlung abgeschlossen.")

    # --- DATEN ZUSAMMENFÜHREN ---
    all_X, all_y = [], []
    for class_id, list_of_arrays in class_features.items():
        if list_of_arrays:
            arr = np.concatenate(list_of_arrays, axis=0)
            all_X.append(arr)
            all_y.append(np.full(len(arr), class_id))

    # --- FEHLER-CHECK: Haben wir überhaupt Daten? ---
    if not all_X:
        print("\n❌ FEHLER: Keine Punkte gefunden!")
        print("Mögliche Ursachen:")
        print("1. Die Klassen-IDs im Datensatz stimmen nicht mit dem Skript überein (0-5).")
        print("   (Schau auf die 'Vorhandene IDs' Ausgabe oben)")
        print("2. Die Validierungsdaten enthalten nur 'Ignore Labels' (oft 255 oder -1).")
        return

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"\n=> Starte t-SNE Berechnung auf {X.shape[0]} Punkten (Feature-Dim: {X.shape[1]})...")
    print("   (Das kann einen Moment dauern...)")
    
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(X)
    except Exception as e:
        print(f"❌ Fehler bei t-SNE Berechnung: {e}")
        return
    
    print("=> Erstelle Plot...")
    plt.figure(figsize=(12, 10))
    
    for class_id in range(len(CLASSES)):
        mask = (y == class_id)
        if np.any(mask):
            plt.scatter(
                X_embedded[mask, 0], 
                X_embedded[mask, 1], 
                c=COLORS[class_id], 
                label=f"{CLASSES[class_id]} ({np.sum(mask)})",
                alpha=0.6, 
                s=20,
                edgecolor='none'
            )

    plt.title(f"Balanced t-SNE: {mode.upper()} Space\n(Target: {POINTS_PER_CLASS} pts/class)", fontsize=15)
    plt.legend(title="Klassen", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    
    save_path = os.path.join(exp_dir, f"tsne_balanced_{mode}_{os.path.basename(exp_dir.rstrip('/'))}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot gespeichert: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", type=str, required=True, help="Pfad zum Experiment Ordner")
    parser.add_argument("-m", "--mode", type=str, default="latent", choices=["latent", "logits", "input"], 
                        help="'latent'=Features, 'logits'=Output, 'input'=Rohdaten")
    args = parser.parse_args()
    
    visualize_latent_space(args.exp_dir, args.mode)