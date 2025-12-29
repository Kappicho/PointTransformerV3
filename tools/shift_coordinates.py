import pandas as pd
import os

# --- KONFIGURATION ---
input_datei = 'Standpunkt2.asc'
output_datei = '.asc'

# Werte, die abgezogen werden sollen (X, Y, Z)
offset_x = 6.642
offset_y = -38.898
offset_z = -0.746

# Wie viele Zeilen pro Durchgang verarbeitet werden (z.B. 100.000)
# Bei sehr wenig RAM diesen Wert verringern.
chunk_size = 500000 

# Trennzeichen in deiner ASC Datei (meistens Leerzeichen ' ' oder Tabulator '\t')
trennzeichen = ' ' 

# Hat die Datei eine Kopfzeile mit Spaltennamen? 
# Setze auf True, wenn die erste Zeile Text ist (z.B. "X Y Z Intensity")
# Setze auf False, wenn es direkt mit Zahlen losgeht.
hat_kopfzeile = False 

# ---------------------

def process_large_asc():
    print(f"Starte Verarbeitung von {input_datei}...")
    
    # Header-Logik für Pandas
    header_arg = 0 if hat_kopfzeile else None
    
    # Datei in Chunks lesen
    reader = pd.read_csv(
        input_datei, 
        sep=trennzeichen, 
        header=header_arg, 
        chunksize=chunk_size, 
        engine='python', # 'python' engine ist flexibler bei Trennzeichen
        skipinitialspace=True # Ignoriert führende Leerzeichen
    )

    first_chunk = True
    
    # Falls die Datei bereits existiert, löschen wir sie vorher, um sauber zu starten
    if os.path.exists(output_datei):
        os.remove(output_datei)

    chunk_count = 0

    for chunk in reader:
        # Die ersten 3 Spalten bearbeiten (Index 0, 1, 2)
        # Wir nutzen iloc, um sicherzustellen, dass wir spaltenbasiert arbeiten
        chunk.iloc[:, 0] = chunk.iloc[:, 0] - offset_x
        chunk.iloc[:, 1] = chunk.iloc[:, 1] - offset_y
        chunk.iloc[:, 2] = chunk.iloc[:, 2] - offset_z
        
        # Schreib-Modus: 'w' (write) beim ersten Chunk, danach 'a' (append)
        mode = 'w' if first_chunk else 'a'
        
        # Header nur beim ersten Chunk schreiben, falls original einer da war
        write_header = hat_kopfzeile if first_chunk else False
        
        # In Zieldatei schreiben
        chunk.to_csv(
            output_datei, 
            mode='a', # Wir nutzen 'append' mode manuell, da wir oben löschen
            sep=trennzeichen, 
            index=False, 
            header=write_header
        )
        
        first_chunk = False
        chunk_count += 1
        print(f"Chunk {chunk_count} verarbeitet...", end='\r')

    print(f"\nFertig! Datei gespeichert unter: {output_datei}")

if __name__ == "__main__":
    process_large_asc()