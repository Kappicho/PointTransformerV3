import torch
import numpy as np
import os
import sys
import argparse
from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.datasets import build_dataset

def load_checkpoint(model, filename):
    print(f"=> Lade Checkpoint: {filename}")
    checkpoint = torch.load(filename, map_location='cuda:0')
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    print("=> Gewichte geladen.")

# --- COLLATE FUNKTION ---
def collate_fn(batch):
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    data_dict = {}
    
    # 1. Normale Felder
    for key in batch[0].keys():
        if key in ["feat", "coord", "grid_coord", "segment"]: 
            continue
        data_dict[key] = [b[key] for b in batch]

    # 2. Koordinaten
    coords_list = [to_tensor(b["coord"]) for b in batch]
    coords_flat = torch.cat(coords_list, dim=0)
    
    # 3. Offset
    offset_list = [c.shape[0] for c in coords_list]
    offset = torch.cumsum(torch.tensor(offset_list), dim=0).int()
    
    # 4. Grid Coords
    grid_list = [to_tensor(b["grid_coord"]) for b in batch]
    grid_flat = torch.cat(grid_list, dim=0)
    
    # 5. Batch Index
    batch_list = []
    for i, g in enumerate(grid_list):
        batch_list.append(torch.full((g.shape[0],), i, dtype=torch.long))
    batch_flat = torch.cat(batch_list, dim=0)

    # 6. Features
    if "feat" in batch[0]:
        feat_flat = torch.cat([to_tensor(b["feat"]) for b in batch], dim=0).float()
    else:
        feat_flat_list = []
        for b in batch:
            f = []
            if "color" in b: f.append(to_tensor(b["color"]))
            if "normal" in b: f.append(to_tensor(b["normal"]))
            if "strength" in b: f.append(to_tensor(b["strength"]))
            if len(f) > 0:
                feat_flat_list.append(torch.cat(f, dim=1))
            else:
                feat_flat_list.append(torch.zeros((b["coord"].shape[0], 1))) 
        feat_flat = torch.cat(feat_flat_list, dim=0).float()
    
    # 7. Labels
    segment_flat = torch.cat([to_tensor(b["segment"]) for b in batch], dim=0).long()

    return {
        "coord": coords_flat,
        "grid_coord": grid_flat,
        "batch": batch_flat,
        "segment": segment_flat,
        "feat": feat_flat,
        "offset": offset,
        "batch_size": len(batch)
    }

def export_to_cloudcompare(exp_dir):
    # Pfade dynamisch zusammenbauen
    config_path = os.path.join(exp_dir, "config.py")
    output_folder = os.path.join(exp_dir, "cloudcompare_export")
    
    # Checkpoints suchen
    best_pth = os.path.join(exp_dir, "model/model_best.pth")
    last_pth = os.path.join(exp_dir, "model/model_last.pth")
    
    if os.path.exists(best_pth):
        checkpoint_path = best_pth
        print(f"=> Nutze bestes Modell: {best_pth}")
    elif os.path.exists(last_pth):
        checkpoint_path = last_pth
        print(f"=> Nutze letztes Modell: {last_pth}")
    else:
        print(f"FEHLER: Kein Modell in {exp_dir} gefunden!")
        return

    os.makedirs(output_folder, exist_ok=True)

    print(f"=> Lade Config: {config_path}")
    if not os.path.exists(config_path):
        print("Config nicht gefunden!")
        return

    cfg = Config.fromfile(config_path)
    
    print(f"=> Baue Modell...")
    model = build_model(cfg.model).cuda()
    load_checkpoint(model, checkpoint_path)
    model.eval()

    print("=> Lade Datensatz...")
    dataset = build_dataset(cfg.data.val)
    
    print(f"=> Starte Export nach: {output_folder}")
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            try:
                try:
                    name = dataset.get_data_name(idx)
                except:
                    name = f"scene_{idx:04d}"
                
                print(f"Verarbeite: {name} ...")

                raw_data = dataset[idx]
                batch_data = collate_fn([raw_data])

                # Alles auf GPU
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)

                # Vorhersage
                output_dict = model(batch_data)
                pred_logits = output_dict["seg_logits"]
                pred_labels = pred_logits.max(1)[1].cpu().numpy()

                # Daten zurückholen
                coord = batch_data["coord"].cpu().numpy()
                gt_labels = batch_data["segment"].cpu().numpy()
                
                feat = batch_data["feat"].cpu().numpy()
                if feat.shape[1] >= 3:
                    color = feat[:, 0:3]
                    if np.max(color) <= 1.5:
                        color *= 255.0
                else:
                    color = np.ones_like(coord) * 128.0

                error_mask = (pred_labels != gt_labels).astype(int)
                
                export_data = np.column_stack((coord, color, gt_labels, pred_labels, error_mask))

                save_path = os.path.join(output_folder, f"{name}_prediction.txt")
                header = "X Y Z R G B GroundTruth Prediction IsError"
                
                np.savetxt(save_path, export_data, fmt='%.4f %.4f %.4f %d %d %d %d %d %d', header=header, comments='')
                print(f" -> OK: {save_path}")

            except Exception as e:
                print(f"FEHLER bei {name}: {e}")
                import traceback
                traceback.print_exc()

    print("-" * 50)
    print("FERTIG! Export abgeschlossen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", type=str, required=True, help="Pfad zum Experiment-Ordner")
    args = parser.parse_args()
    
    export_to_cloudcompare(args.exp_dir)
