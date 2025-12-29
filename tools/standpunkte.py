import sys
import subprocess

# --- 1. Automatische Installation von Pandas prüfen ---
try:
    import pandas as pd
except ImportError:
    print("Pandas wurde nicht gefunden. Installiere es jetzt automatisch...")
    # Führt 'pip install pandas' innerhalb des Skripts aus
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd
    print("Pandas erfolgreich installiert.")

import os

def vergleiche_punktwolken(datei1_pfad, datei2_pfad, ausgabe_pfad):
    """
    Vergleicht zwei Punktwolken-Dateien anhand der Koordinaten x, y, z.
    Exportiert Zeilen aus Datei 2, die auch in Datei 1 vorkommen.
    """
    
    # Spaltennamen definieren
    # Datei 1: x,y,z,Intensität,r,g,b,Nx,Ny,Nz
    spalten_1 = ['x', 'y', 'z', 'I', 'r', 'g', 'b', 'Nx', 'Ny', 'Nz']
    
    # Datei 2: x,y,z,Intensität,klasse,r,g,b,Nx,Ny,Nz
    spalten_2 = ['x', 'y', 'z', 'I', 'klasse', 'r', 'g', 'b', 'Nx', 'Ny', 'Nz']

    print("Lade Dateien ein...")

    # sep='\s+' behandelt ein oder mehrere Leerzeichen als Trenner
    # engine='python' ist stabiler
    df1 = pd.read_csv(datei1_pfad, sep='\s+', names=spalten_1, header=None, engine='python')
    df2 = pd.read_csv(datei2_pfad, sep='\s+', names=spalten_2, header=None, engine='python')

    print(f"Datei 1 ({len(df1)} Zeilen) geladen.")
    print(f"Datei 2 ({len(df2)} Zeilen) geladen.")

    # --- WICHTIG: Runden auf 3 Nachkommastellen ---
    # Damit wird sichergestellt, dass Floating-Point-Ungenauigkeiten 
    # nicht dazu führen, dass gleiche Punkte als unterschiedlich erkannt werden.
    df1['x'] = df1['x'].round(3)
    df1['y'] = df1['y'].round(3)
    df1['z'] = df1['z'].round(3)

    df2['x'] = df2['x'].round(3)
    df2['y'] = df2['y'].round(3)
    df2['z'] = df2['z'].round(3)

    # Wir brauchen aus Datei 1 nur die eindeutigen Koordinaten für den Vergleich
    # Das spart Speicher und beschleunigt den Merge
    referenz_koordinaten = df1[['x', 'y', 'z']].drop_duplicates()

    print("Starte Abgleich (Inner Join)...")
    
    # Merge durchführen: Behält nur Zeilen in df2, deren x,y,z in referenz_koordinaten vorkommen
    ergebnis = pd.merge(df2, referenz_koordinaten, on=['x', 'y', 'z'], how='inner')

    print(f"Anzahl übereinstimmender Zeilen: {len(ergebnis)}")

    # Exportieren
    # sep=' ' sorgt dafür, dass die Ausgabedatei auch Leerzeichen als Trenner hat
    print(f"Schreibe Ergebnisse in '{ausgabe_pfad}'...")
    ergebnis.to_csv(ausgabe_pfad, sep=' ', index=False, header=False)
    
    print("Fertig!")

# --- KONFIGURATION ---
# Bitte hier die Dateinamen anpassen oder absolute Pfade eintragen
input_datei_1 = 'Standpunkt1.asc'
input_datei_2 = 'Building_9_1.asc'
output_datei  = 'Building_9_1_1.asc'

if __name__ == "__main__":
    # Prüfen, ob die Dateien existieren, bevor wir starten
    if not os.path.exists(input_datei_1):
        print(f"FEHLER: Datei '{input_datei_1}' nicht gefunden.")
    elif not os.path.exists(input_datei_2):
        print(f"FEHLER: Datei '{input_datei_2}' nicht gefunden.")
    else:
        vergleiche_punktwolken(input_datei_1, input_datei_2, output_datei)