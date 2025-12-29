import os

target_file = "pointcept/engines/train.py"
backup_file = "pointcept/engines/train.py.bak"

# Backup erstellen
if not os.path.exists(backup_file):
    os.system(f"cp {target_file} {backup_file}")
    print("Backup erstellt.")

with open(target_file, "r") as f:
    lines = f.readlines()

new_lines = []
inserted = False

for line in lines:
    new_lines.append(line)
    # Wir suchen den Start der Trainings-Schleife
    if "for i, input_dict in enumerate(self.train_loader):" in line and not inserted:
        indent = line.split("for")[0] + "    " # Einrückung ermitteln (meist 8 oder 12 spaces)
        
        # Unser Wächter-Code
        guard_code = [
            f"{indent}# --- WATCHDOG START ---\n",
            f"{indent}if 'segment' in input_dict:\n",
            f"{indent}    lbl = input_dict['segment']\n",
            f"{indent}    if lbl.max() >= 9:\n",
            f"{indent}        print(f'\\n🚨 ALARM in Batch {{i}}! Ungültiges Label gefunden: {{lbl.max().item()}}')\n",
            f"{indent}        if 'name' in input_dict: print(f'Schuldige Datei(en): {{input_dict[\"name\"]}}')\n",
            f"{indent}        import sys; sys.exit(1)\n",
            f"{indent}    if (lbl < -1).any():\n",
            f"{indent}        print(f'\\n🚨 ALARM! Negatives Label (außer -1) gefunden!')\n",
            f"{indent}        import sys; sys.exit(1)\n",
            f"{indent}# --- WATCHDOG END ---\n"
        ]
        new_lines.extend(guard_code)
        inserted = True
        print("Watchdog erfolgreich eingefügt!")

if inserted:
    with open(target_file, "w") as f:
        f.writelines(new_lines)
    print(f"Datei {target_file} wurde aktualisiert.")
else:
    print("FEHLER: Konnte die Einfüge-Stelle nicht finden.")