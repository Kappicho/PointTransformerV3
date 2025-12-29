import os

# Datei Pfad
train_file = "pointcept/engines/train.py"

# Backup
if not os.path.exists(train_file + ".bak"):
    os.system(f"cp {train_file} {train_file}.bak")

with open(train_file, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Wir suchen den Start des Forward Pass
    if "output_dict = self.model(input_dict)" in line:
        indent = line.split("output_dict")[0]
        # Debug Code einfügen
        new_lines.append(f"{indent}# --- DEBUG WATCHDOG ---\n")
        new_lines.append(f"{indent}if 'coord' in input_dict and input_dict['coord'].isnan().any():\n")
        new_lines.append(f"{indent}    print(f'ALARM: NaN in Coords! Datei: {{input_dict.get(\"name\", \"Unbekannt\")}}')\n")
        new_lines.append(f"{indent}if 'segment' in input_dict and (input_dict['segment'].max() >= 9):\n")
        new_lines.append(f"{indent}    print(f'ALARM: Label {{input_dict[\"segment\"].max()}} zu gross! Datei: {{input_dict.get(\"name\", \"Unbekannt\")}}')\n")
        new_lines.append(f"{indent}# ----------------------\n")
    new_lines.append(line)

with open(train_file, 'w') as f:
    f.writelines(new_lines)

print("Watchdog installiert!")