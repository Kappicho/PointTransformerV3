#!/bin/bash

# --- SICHERHEITS-CHECK ---
# Wir setzen den Python-Pfad, damit 'import pointcept' in den Tools funktioniert
export PYTHONPATH=$PYTHONPATH:$(pwd)

# --- 0. ARGUMENTE PARSEN ---
while getopts ":p:g:d:c:n:w:r:" opt; do
  case $opt in
    n) EXP_NAME="$OPTARG"
    ;;
    d) DATASET="$OPTARG"
    ;;
    *) ;; 
  esac
done

if [ -z "$EXP_NAME" ]; then
    echo "FEHLER: Der Name fehlt! Nutzen Sie: sh full_pipeline.sh ... -n run_name"
    exit 1
fi

DATASET=${DATASET:-fassade}
EXP_DIR="exp/${DATASET}/${EXP_NAME}"

echo "=========================================="
echo "   STARTING PIPELINE: $EXP_NAME"
echo "=========================================="

# --- 1. PREPROCESSING ---
echo ""
echo "[1/6] CHECKING DATA..."
python tools/preprocess_and_split.py

# --- 2. TRAINING ---
echo ""
echo "[2/6] STARTING TRAINING..."
# Wir rufen train.sh auf (Argumente werden durchgereicht)
sh scripts/train.sh "$@"

# Checken ob Training erfolgreich war (Modell da?)
if [ ! -f "$EXP_DIR/model/model_last.pth" ]; then
    echo "❌ FEHLER: Training scheint abgebrochen zu sein (Kein Checkpoint gefunden)."
    # Wir machen NICHT weiter, um Folgefehler zu vermeiden
    exit 1
fi

# --- 3. TESTING ---
echo ""
echo "[3/6] TESTING MODEL..."

if [ -f "$EXP_DIR/model/model_best.pth" ]; then
    WEIGHTS="$EXP_DIR/model/model_best.pth"
    echo "-> Nutze 'model_best.pth'"
else
    WEIGHTS="$EXP_DIR/model/model_last.pth"
    echo "-> Nutze 'model_last.pth' (Fallback)"
fi

# Test starten
sh scripts/test.sh -p python -g 1 -d $DATASET -n $EXP_NAME -w "$WEIGHTS"

# --- 4. EXPORT (TXT) ---
if [ -d "$EXP_DIR" ]; then
    echo ""
    echo "[4/6] EXPORTING PREDICTIONS TO TXT..."
    python tools/export_results.py -e "$EXP_DIR"

    # --- 5. REPORT (CSV) ---
    echo ""
    echo "[5/6] GENERATING REPORT..."
    python tools/report_csv.py -e "$EXP_DIR"
    
    # --- 6. CONFUSION MATRIX (NEU) ---
    echo ""
    echo "[6/6] CREATING CONFUSION MATRIX..."
    # Prüfen, ob das Export-Verzeichnis existiert, sonst macht das Skript keinen Sinn
    if [ -d "$EXP_DIR/cloudcompare_export" ]; then
        python tools/confusionmatrix.py -e "$EXP_DIR"
    else
        echo "WARNUNG: Export-Ordner fehlt. Überspringe Matrix."
    fi
    
    echo ""
    echo "✅ PIPELINE SUCCESSFUL!"
    echo "   -> Results: $EXP_DIR/cloudcompare_export"
    echo "   -> Matrix:  $EXP_DIR/confusion_matrix_${EXP_NAME}.png"
    echo "   -> Report:  EXPERIMENTS_SUMMARY.csv"
else
    echo ""
    echo "❌ PIPELINE FAILED (Ordner $EXP_DIR nicht gefunden)"
fi