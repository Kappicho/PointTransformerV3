import os
import sys

# Datei-Pfade
target_file = "pointcept/engines/train.py"
backup_file = "pointcept/engines/train.py.bak"

# Sicherstellen, dass wir im richtigen Ordner sind
if not os.path.exists(target_file):
    print(f"FEHLER: Datei {target_file} nicht gefunden!")
    print("Bitte führen Sie das Skript im Hauptordner 'Pointcept' aus.")
    sys.exit(1)

# Backup erstellen (nur einmal)
if not os.path.exists(backup_file):
    os.system(f"cp {target_file} {backup_file}")
    print(f"Backup erstellt: {backup_file}")

with open(target_file, "r") as f:
    lines = f.readlines()

new_lines = []
inserted = False
watchdog_marker = "# --- WATCHDOG START ---"

# Checken, ob schon gepatcht
for line in lines:
    if watchdog_marker in line:
        print("Datei ist bereits gepatcht! Breche ab.")
        sys.exit(0)

# Patchen
for line in lines:
    # Wir suchen den Aufruf der Trainings-Funktion
    if "self.run_step()" in line and not inserted:
        # Einrückung der aktuellen Zeile übernehmen
        indent = line.split("self.run_step()")[0]
        
        # Der Watchdog Code
        code = [
            f"{indent}{watchdog_marker}\n",
            f"{indent}# Automatische Prüfung auf korrupte Daten\n",
            f"{indent}if 'segment' in input_dict:\n",
            f"{indent}    _lbl = input_dict['segment']\n",
            f"{indent}    # Ignoriere -1 (Unlabeled)\n",
            f"{indent}    _valid_lbl = _lbl[_lbl != -1]\n",
            f"{indent}    if _valid_lbl.numel() > 0:\n",
            f"{indent}        _max_l = _valid_lbl.max().item()\n",
            f"{indent}        _min_l = _valid_lbl.min().item()\n",
            f"{indent}        if _max_l >= 9:\n",
            f"{indent}            print(f'\\n🚨 ALARM! Label {{_max_l}} gefunden (Erlaubt: 0-8)!')\n",
            f"{indent}            if 'name' in input_dict: print(f'DATEI: {{input_dict[\"name\"]}}')\n",
            f"{indent}            import sys; sys.exit(1)\n",
            f"{indent}if 'coord' in input_dict:\n",
            f"{indent}    if input_dict['coord'].isnan().any():\n",
            f"{indent}        print(f'\\n🚨 ALARM! NaN in Koordinaten!')\n",
            f"{indent}        if 'name' in input_dict: print(f'DATEI: {{input_dict[\"name\"]}}')\n",
            f"{indent}        import sys; sys.exit(1)\n",
            f"{indent}# --- WATCHDOG END ---\n"
        ]
        new_lines.extend(code)
        inserted = True
    
    new_lines.append(line)

if inserted:
    with open(target_file, "w") as f:
        f.writelines(new_lines)
    print("✅ Watchdog erfolgreich vor 'self.run_step()' eingefügt.")
else:
    print("❌ FEHLER: Konnte 'self.run_step()' nicht finden. Code wurde nicht geändert.")
