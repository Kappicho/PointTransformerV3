import os

# Wir patchen direkt den Code im laufenden Experiment
target_file = "exp/fassade/run04/code/pointcept/engines/train.py"

if not os.path.exists(target_file):
    print(f"FEHLER: Datei nicht gefunden: {target_file}")
    exit(1)

print(f"Patche Datei: {target_file}")

with open(target_file, "r") as f:
    lines = f.readlines()

new_lines = []
inserted = False

watchdog_code = [
    "                # --- WATCHDOG START ---\n",
    "                if 'coord' in input_dict:\n",
    "                    _c = input_dict['coord']\n",
    "                    if _c.abs().max() > 10000:\n",
    "                        print(f'\\n🚨 ALARM! Extreme Koordinaten ({_c.abs().max()}) gefunden!')\n",
    "                        if 'name' in input_dict: print(f'DATEI: {input_dict[\"name\"]}')\n",
    "                        import sys; sys.exit(1)\n",
    "                if 'segment' in input_dict:\n",
    "                    lbl = input_dict['segment']\n",
    "                    valid_lbl = lbl[lbl != -1]\n",
    "                    if valid_lbl.numel() > 0 and valid_lbl.max() >= 9:\n",
    "                        print(f'\\n🚨 ALARM! Label {valid_lbl.max()} gefunden!')\n",
    "                        if 'name' in input_dict: print(f'DATEI: {input_dict[\"name\"]}')\n",
    "                        import sys; sys.exit(1)\n",
    "                # --- WATCHDOG END ---\n"
]

for line in lines:
    # Wir fügen den Check direkt VOR der Ausführung des Modells ein
    if "self.run_step()" in line and not inserted:
        # Einrückung der aktuellen Zeile übernehmen (Leerzeichen am Anfang)
        indent = line[:len(line) - len(line.lstrip())]
        
        # Code einfügen
        for code_line in watchdog_code:
            new_lines.append(indent + code_line.strip() + "\n")
        
        inserted = True
        print("Watchdog vor 'self.run_step()' eingefügt.")
    
    new_lines.append(line)

if inserted:
    with open(target_file, "w") as f:
        f.writelines(new_lines)
    print("✅ Erfolg! Der Wächter ist aktiv.")
else:
    print("❌ FEHLER: Konnte die Stelle 'self.run_step()' nicht finden.")
