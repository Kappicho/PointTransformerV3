import os

def save_structure_to_txt(startpath, filename="project_structure.txt"):
    # Datei zum Schreiben öffnen
    with open(filename, "w", encoding="utf-8") as f:
        # Root-Verzeichnis schreiben
        root_name = os.path.basename(os.path.abspath(startpath)) or startpath
        f.write(f"📂 {root_name}\n")
        
        for root, dirs, files in os.walk(startpath):
            # 1. Filter: Unwichtige System- und Cache-Ordner ignorieren
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.ipynb_checkpoints', 'wandb', 'logs', 'data']]
            
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            
            # Ordnernamen schreiben (außer Startordner, der steht schon oben)
            if root != startpath:
                f.write(f"{indent}├── 📁 {os.path.basename(root)}/\n")
            
            subindent = ' ' * 4 * (level + 1)
            
            for file in files:
                # 2. Filter: Nur relevante Code- und Konfigurationsdateien auflisten
                if file.endswith(('.py', '.ipynb', '.yaml', '.yml', '.json', '.sh', '.md', '.txt')):
                    f.write(f"{subindent}├── {file}\n")
                    
    print(f"✅ Fertig! Die Struktur wurde in '{filename}' gespeichert.")

# Skript ausführen ('.' steht für den aktuellen Ordner)
save_structure_to_txt('.')