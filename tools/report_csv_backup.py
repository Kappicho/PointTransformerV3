import os
import sys
import glob
import re
import csv
import argparse
from datetime import datetime

# Import-Fix für Pointcept Config
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pointcept.utils.config import Config

MASTER_CSV_PATH = "EXPERIMENTS_SUMMARY.csv"
CSV_DELIMITER = ';' # Semikolon für deutsches Excel

# --- FESTE SPALTEN DEFINITION ---
KNOWN_CLASSES = [
    "Sonstiges", "Putz", "Beton", "Backstein", "Dachziegel", 
    "SteinNatur", "SteinFliesen", "Kunststoff", "Metall"
]

def get_data_list_string(root_path, split_name):
    """Liest die sauberen Dateinamen aus meta/split.txt."""
    if not root_path or root_path == "N/A": return "N/A"
    
    list_path = os.path.join(root_path, "meta", f"{split_name}.txt")
    if os.path.exists(list_path):
        with open(list_path, 'r') as f:
            files = [l.strip() for l in f.readlines() if l.strip()]
        count = len(files)
        file_str = " | ".join(files)
        if len(file_str) > 100:
            return f"{count} Files: " + " | ".join(files[:3]) + " ..."
        return file_str
    return "N/A"

def calculate_loop_factor_from_log(log_path, root_path):
    """Berechnet Loop Factor historisch korrekt aus dem Logfile."""
    # 1. Basis-Anzahl aus meta/train.txt ermitteln
    list_path = os.path.join(root_path, "meta", "train.txt")
    if not os.path.exists(list_path): return "N/A (Meta missing)"
    
    with open(list_path, 'r') as f:
        base_count = len([l for l in f.readlines() if l.strip()])
    
    if base_count == 0: return "0"

    # 2. Tatsächliche Anzahl aus dem Log parsen
    if not log_path or not os.path.exists(log_path): return "N/A (No Log)"
    
    total_samples = 0
    found = False
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "Totally" in line and "samples in" in line and "train set" in line:
                try:
                    parts = line.split("Totally")[1].strip().split(" ")
                    total_samples = int(parts[0])
                    found = True
                    break 
                except: pass
    
    if not found: return "N/A (Not in Log)"
    
    # 3. Faktor berechnen
    factor = total_samples / base_count
    return f"{factor:.1f}x ({total_samples}/{base_count})"

def get_log_metrics(log_path):
    metrics = {}
    if not os.path.exists(log_path): return metrics
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f: 
        lines = f.readlines()
    
    if not lines: return metrics

    best_miou = -1.0
    current_block = {}
    best_block = {}
    
    # Variablen für Metriken
    max_epoch = 0
    start_time = None
    end_time = None
    final_val_loss = "N/A"
    
    # Liste zum Sammeln der Losses einer Validierungs-Runde
    current_val_losses = [] 
    
    ts_pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

    for line in lines:
        # --- ZEIT & DAUER ---
        match = ts_pattern.search(line)
        if match:
            try:
                current_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                if start_time is None: start_time = current_time
                end_time = current_time
            except: pass

        # --- EPOCHEN FORTSCHRITT ---
        if "Train: [" in line:
            try:
                part = line.split("Train: [")[1].split("]")[0]
                ep = int(part.split("/")[0])
                if ep > max_epoch: max_epoch = ep
            except: pass

        # --- VALIDIERUNGS LOSS SAMMELN (NEU) ---
        # Sucht nach Zeilen wie: "Test: [ 1/12] ... Loss 1.2345"
        if "Test: [" in line and "Loss" in line:
            try:
                # Extrahiere den Wert nach "Loss"
                loss_part = line.split("Loss")[1].strip().split()[0]
                val_loss_float = float(loss_part)
                current_val_losses.append(val_loss_float)
            except: pass

        # --- VALIDIERUNGS ERGEBNISSE (MEAN BERECHNEN) ---
        if "Val result: mIoU/mAcc/allAcc" in line:
            # Jetzt berechnen wir den Durchschnitt der gesammelten Losses für diese Epoche
            if current_val_losses:
                mean_loss = sum(current_val_losses) / len(current_val_losses)
                val_loss_str = f"{mean_loss:.5f}" # Auf 5 Nachkommastellen formatieren
                
                # Speichern für den aktuellen Block und global
                final_val_loss = val_loss_str
                current_block["Final Val Loss"] = val_loss_str
                
                # Liste leeren für nächste Runde
                current_val_losses = []
            
            try:
                vals = line.split("mIoU/mAcc/allAcc")[1].strip().rstrip('.').split('/')
                miou = float(vals[0])
                current_block["mIoU"] = vals[0]
                current_block["mAcc"] = vals[1]
                current_block["allAcc"] = vals[2]
                
                if miou >= best_miou:
                    best_miou = miou
                    best_block = current_block.copy()
            except: pass

        # --- KLASSEN ERGEBNISSE ---
        if "Result: iou/accuracy" in line:
            try:
                parts = line.split("Result: iou/accuracy")
                prefix = parts[0].strip()
                raw_name_full = prefix.split(" ")[-1]
                
                if "Class_" in raw_name_full and "-" in raw_name_full:
                    clean_name = raw_name_full.split("-", 1)[1]
                else:
                    clean_name = raw_name_full

                scores = parts[1].strip().split('/')
                current_block[f"IoU_{clean_name}"] = scores[0]
                current_block[f"Acc_{clean_name}"] = scores[1]
                
                # Update best_block falls dies zur besten Epoche gehört
                if current_block.get("mIoU") == best_block.get("mIoU"):
                    best_block.update(current_block)
            except: pass
                
        if "Train result: loss:" in line:
            best_block["Final Train Loss"] = line.split("loss:")[1].strip()
        if "Max Mem:" in line:
            best_block["Max Memory"] = line.split("Max Mem:")[1].strip()
    
    # --- ZUSAMMENFASSUNG ---
    if max_epoch > 0:
        best_block["Epochs Reached"] = max_epoch
    
    # Falls 'Final Val Loss' noch nicht im Best Block ist (z.B. Abbruch vor Val Result), den letzten bekannten nehmen
    if "Final Val Loss" not in best_block:
        best_block["Final Val Loss"] = final_val_loss
    
    if start_time and end_time:
        duration = end_time - start_time
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            duration_str = f"{hours}h {minutes}m"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{seconds}s"
        
        best_block["Training Duration"] = duration_str

    return best_block

def get_config_details(config_path, log_path=None):
    info = {}
    if not os.path.exists(config_path): return info
    try:
        cfg = Config.fromfile(config_path)
        info["Model Type"] = cfg.model.get('type', 'N/A')
        if 'backbone' in cfg.model:
            info["Backbone"] = cfg.model.backbone.get('type', 'N/A')
            info["Backbone Channels"] = cfg.model.get('backbone_out_channels', 'N/A')
        info["Pretrained"] = cfg.get('load_from', 'None')
        if 'data' in cfg:
            info["Num Classes"] = cfg.data.get('num_classes', 'N/A')
            info["Data Root"] = cfg.get('data_root', cfg.data.get('data_root', 'N/A'))
        
        if "Data Root" in info and info["Data Root"] != "N/A":
            info["Train Files"] = get_data_list_string(info["Data Root"], "train")
            info["Val Files"] = get_data_list_string(info["Data Root"], "val")
            info["Loop Factor"] = calculate_loop_factor_from_log(log_path, info["Data Root"])
            
        info["Epochs"] = cfg.get('epoch', 'N/A')
        info["Batch Size"] = cfg.get('batch_size', 'N/A')
        info["Num Worker"] = cfg.get('num_worker', 'N/A')
        info["AMP"] = cfg.get('enable_amp', 'N/A')
        if 'optimizer' in cfg:
            info["Optimizer"] = cfg.optimizer.get('type', 'N/A')
            info["LR"] = cfg.optimizer.get('lr', 'N/A')
            info["Weight Decay"] = cfg.optimizer.get('weight_decay', 'N/A')
        if 'scheduler' in cfg:
            info["Scheduler"] = cfg.scheduler.get('type', 'N/A')
        
        if 'model' in cfg and 'criteria' in cfg.model:
            criteria = cfg.model.criteria
            if not isinstance(criteria, list): criteria = [criteria]
            for crit in criteria:
                c_type = crit.get('type', '')
                if 'CrossEntropy' in c_type:
                    info["Loss Weight (CE)"] = crit.get('loss_weight', 'N/A')
                    cw = crit.get('weight')
                    if cw is not None: info["Class Weights"] = str(cw)
                if 'Lovasz' in c_type:
                    info["Loss Weight (Lovasz)"] = crit.get('loss_weight', 'N/A')

        aug_list = []
        if 'data' in cfg and 'train' in cfg.data and 'transform' in cfg.data.train:
            transforms = cfg.data.train.transform
            for t in transforms:
                name = t['type']
                if name in ["ToTensor", "Collect", "NormalizeColor", "CenterShift"]: continue
                details = ""
                if name == "GridSample":
                    info["Voxel Size"] = t.get('grid_size', 'N/A')
                    continue
                if name == "SphereCrop":
                    info["SphereCrop Points"] = t.get('point_max', 'N/A')
                    info["SphereCrop Mode"] = t.get('mode', 'N/A')
                    continue
                if name == "RandomRotate": details = f"angle={t.get('angle')}"
                elif name == "RandomScale": details = f"scale={t.get('scale')}"
                elif name == "RandomDrop": details = f"p={t.get('p')}"
                elif name == "RandomJitter": details = f"sigma={t.get('sigma')}"
                if details: aug_list.append(f"{name}({details})")
                else: aug_list.append(name)
            for t in transforms:
                if t['type'] == 'Collect':
                    info["Input Features"] = "+".join(t.get('feat_keys', []))
                    break
        info["Augmentations"] = " + ".join(aug_list)
    except Exception as e:
        print(f"Warnung Config: {e}")
    return info

def update_csv(exp_dir):
    run_name = os.path.basename(exp_dir.rstrip('/'))
    config_path = os.path.join(exp_dir, "config.py")
    
    train_log = os.path.join(exp_dir, "train.log")
    log_files = glob.glob(os.path.join(exp_dir, "*.log"))
    
    if os.path.exists(train_log):
        log_path = train_log
    elif log_files:
        log_path = max(log_files, key=os.path.getmtime)
    else:
        print("Kein Log gefunden.")
        return

    print(f"Sammle Daten für: {run_name} (Quelle: {os.path.basename(log_path)}) ...")
    row_data = {
        "Run Name": run_name,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Log File": os.path.basename(log_path)
    }
    
    row_data.update(get_config_details(config_path, log_path))
    row_data.update(get_log_metrics(log_path))

    all_rows = []
    file_exists = os.path.isfile(MASTER_CSV_PATH)
    fieldnames = set(row_data.keys())
    
    for cls in KNOWN_CLASSES:
        fieldnames.add(f"IoU_{cls}")
        fieldnames.add(f"Acc_{cls}")

    if file_exists:
        with open(MASTER_CSV_PATH, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=CSV_DELIMITER)
            fieldnames.update(reader.fieldnames)
            for row in reader:
                if row["Run Name"] != run_name:
                    all_rows.append(row)
    
    all_rows.append(row_data)
    all_rows.sort(key=lambda x: x.get("Run Name", ""))

    priority = [
        "Run Name","Timestamp", "mIoU", "mAcc", "allAcc", "Final Train Loss", "Final Val Loss","Epochs",
        "Epochs Reached", "Training Duration", 
        "Loss Weight (CE)", "Loss Weight (Lovasz)", "Class Weights",
        "Loop Factor", "Voxel Size", "SphereCrop Points", "Augmentations", 
        "Input Features", "Pretrained", "Num Classes",
        "LR", "Batch Size", "Optimizer", 
        "Train Files", "Val Files", "Model Type", "Backbone", 
    ]
    
    sorted_fieldnames = [f for f in priority if f in fieldnames]
    other_fields = sorted(list(fieldnames - set(sorted_fieldnames)))
    
    class_iou = sorted([f for f in other_fields if f.startswith("IoU_")])
    class_acc = sorted([f for f in other_fields if f.startswith("Acc_")])
    misc = sorted([f for f in other_fields if f not in class_iou and f not in class_acc])
    
    final_header = sorted_fieldnames + class_iou + class_acc + misc

    with open(MASTER_CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_header, delimiter=CSV_DELIMITER)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✅ Report aktualisiert: {MASTER_CSV_PATH} (Sortiert)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", type=str, required=True)
    args = parser.parse_args()
    update_csv(args.exp_dir)