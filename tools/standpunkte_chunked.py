import sys
import subprocess
import os
import time

# --- Automatische Installation von Pandas prüfen ---
try:
    import pandas as pd
except ImportError:
    print("Pandas wird installiert...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

def vergleiche_punktwolken_chunked(datei1_pfad, datei2_pfad, ausgabe_pfad):
    start_zeit = time.time()
    
    # Spaltennamen definieren
    spalten_1 = ['x', 'y', 'z', 'I', 'r', 'g', 'b', 'Nx', 'Ny', 'Nz']
    spalten_2 = ['x', 'y', 'z', 'I', 'klasse', 'r', 'g', 'b', 'Nx', 'Ny', 'Nz']

    # --- SCHRITT 1: Datei 1 (Referenz) laden ---
    # Da Datei 1 jetzt < 100MB ist, laden wir sie komplett.
    print(f"Lade Referenz-Datei 1: {datei1_pfad}")
    
    # Wir laden NUR die Koordinaten aus Datei 1, das spart Speicher
    df1 = pd.read_csv(datei1_pfad, sep='\s+', names=spalten_1, header=None, engine='python', usecols=[0, 1, 2])
    
    # Runden und Duplikate entfernen
    df1['x'] = df1['x'].round(3)
    df1['y'] = df1['y'].round(3)
    df1['z'] = df1['z'].round(3)
    df1.drop_duplicates(inplace=True)
    
    print(f"-> Referenz geladen: {len(df1)} eindeutige Punkte.")

    # --- SCHRITT 2: Output-Datei vorbereiten ---
    # Falls die Datei schon existiert, löschen wir sie, da wir gleich "anhängen" (append)
    if os.path.exists(ausgabe_pfad):
        os.remove(ausgabe_pfad)

    # --- SCHRITT 3: Datei 2 in Stücken (Chunks) verarbeiten ---
    print(f"Starte blockweise Verarbeitung von Datei 2: {datei2_pfad}")
    
    # Chunksize: Anzahl der Zeilen pro Durchlauf. 
    # 500.000 Zeilen verbrauchen relativ wenig RAM, sind aber performant.
    chunk_size = 500000 
    total_treffer = 0
    chunk_counter = 0

    # Wir lesen Datei 2 stückweise
    reader = pd.read_csv(datei2_pfad, sep='\s+', names=spalten_2, header=None, engine='python', chunksize=chunk_size)

    for chunk in reader:
        chunk_counter += 1
        
        # Runden der aktuellen Koordinaten im Chunk
        chunk['x'] = chunk['x'].round(3)
        chunk['y'] = chunk['y'].round(3)
        chunk['z'] = chunk['z'].round(3)

        # Der Abgleich (Inner Join)
        # merge schaut, welche Zeilen aus 'chunk' auch in 'df1' sind
        ergebnis_chunk = pd.merge(chunk, df1, on=['x', 'y', 'z'], how='inner')
        
        anzahl = len(ergebnis_chunk)
        total_treffer += anzahl

        if anzahl > 0:
            # Ergebnis sofort in die Datei schreiben (mode='a' bedeutet append/anhängen)
            ergebnis_chunk.to_csv(ausgabe_pfad, sep=' ', index=False, header=False, mode='a')
        
        print(f"   Block {chunk_counter} verarbeitet. Treffer in diesem Block: {anzahl}")

    gesamt_dauer = time.time() - start_zeit
    print(f"--- FERTIG ---")
    print(f"Gesamt gefundene Zeilen: {total_treffer}")
    print(f"Dauer: {gesamt_dauer:.2f} Sekunden")

# --- KONFIGURATION ---
input_datei_1 = 'Standpunkt4.asc'   # Die kleine Datei (<100 MB)
input_datei_2 = 'Building_4_1_gt.txt'   # Die große Datei (600 MB)
output_datei  = 'Building_4_1_1_gt.txt'

if __name__ == "__main__":
    if not os.path.exists(input_datei_1):
        print(f"FEHLER: Datei '{input_datei_1}' nicht gefunden.")
    elif not os.path.exists(input_datei_2):
        print(f"FEHLER: Datei '{input_datei_2}' nicht gefunden.")
    else:
        vergleiche_punktwolken_chunked(input_datei_1, input_datei_2, output_datei)