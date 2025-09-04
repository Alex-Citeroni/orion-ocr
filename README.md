# OCR ROI + InternVL3-9B

Ritaglio automatico della regione di testo (ROI), miglioramento immagine e OCR con un VLM (InternVL3-9B) guidato da prompt. Il flusso usa EasyOCR per individuare il testo e scegliere l’orientamento migliore, poi applica CLAHE, upscaling e deskew, quindi passa la ROI al modello vision‑language per la trascrizione.

---

## Caratteristiche

* **Auto‑orientamento** dell’immagine in base a uno **score OCR** (più box, più confidenza, più area → punteggio maggiore).
* **Ritaglio ROI** robusto unendo i box rilevati (dilatazione + componente con area massima).
* **Enhance**: CLAHE sul canale Y, **upscale 2×** e **deskew ±20°**.
* **OCR tramite VLM** (InternVL3‑9B) con **prompt rigido** e **allowlist** di caratteri.
* **Batch** su una cartella di input, salvataggio **ROI** e **testo** per ogni immagine.

---

## 0) Prerequisiti

* **Git** installato (verifica: `git --version`)
* **Python 3.11 o 3.12** (verifica: `python --version` oppure `py --version` su Windows)
* **NVIDIA GPU consigliata** (verifica: `nvidia-smi`). Funziona anche su CPU (lento).

---

## 1) Clona il repository

Scegli HTTPS o SSH e sostituisci `<REPO_URL>`/`<FOLDER>` con i tuoi valori.

**HTTPS**

```bash
git clone https://github.com/Alex-Citeroni/orion-ocr.git <FOLDER>
cd <FOLDER>
```

**SSH**

```bash
git clone git@github.com:Alex-Citeroni/orion-ocr.git <FOLDER>
cd <FOLDER>
```

---

## 2) Crea e attiva l'ambiente

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv   # oppure: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS (bash)

```bash
python3 -m venv .venv    # assicurati che punti a Python 3.11/3.12
source .venv/bin/activate
```

---

## 3) Installa le dipendenze

```powershell
pip install -r requirements.txt
```

## 4) Prepara la cartella di Input

```bash
mkdir -p input
```

> Metti le immagini in `./input`. I risultati andranno in `./output`.

---

## 5) Configurazione rapida (opzionale)

Nel file Python (sezione `CONFIG`):

* `MODEL_ID = "OpenGVLab/InternVL3-9B"`
* `IN_DIR = Path("input")`, `OUT_DIR = Path("output")`
* `ALLOWLIST` caratteri ammessi
* `ROT_ANGLES = [0, 90, 180, 270]`
* `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`

Per EasyOCR, lingua default inglese: `Reader(["en"], gpu=...)`. Aggiungi ad es. `"it"` per l'italiano.

---

## 6) Esecuzione

```bash
python main.py
```

Output atteso:

```
[1/12] IMG_0001.jpg
  ➤ rot(img)=90°, rot(roi)=0°  |  saved: IMG_0001_roi.jpg
Done. Output in: ./output
```

---

## Come funziona (pipeline)

1. **Lettura immagine + EXIF transpose** → BGR.
2. **Auto‑orientamento**: prova gli angoli in `ROT_ANGLES`, valuta lo **score OCR**:

   * Esegue `reader.readtext` con `allowlist` per ottenere (box, testo, confidenza).
   * Somma: `score += max(conf,0) * (len(text)+1) * log10(10 + area_del_box)`.
   * Mantiene l’angolo con punteggio più alto.
3. **Segmentazione ROI**:

   * Filtra box con confidenza ≥ 0.3 e area > 150.
   * Riempie una maschera, la **dilata** con kernel in funzione dell’altezza mediana dei box.
   * Prende la **componente con area massima** e ritaglia con **padding**.
4. **Enhance + Deskew**:

   * CLAHE sul canale Y (spazio YUV), **upscale 2×** bicubico.
   * **Deskew** tramite `minAreaRect`, clamp dell’angolo a ±20°.
5. **Secondo auto‑orientamento** sulla ROI.
6. **OCR VLM**: preprocess 448×448 + normalizzazione, prompt rigido (plain text, allowed chars, righe dall’alto verso il basso).
7. **Salvataggio** ROI e testo.

---

## Personalizzazioni utili

* **Caratteri**: modifica `ALLOWLIST` per includere minuscole o simboli aggiuntivi.
* **Lingue OCR**: cambia `Reader(["en"])`.
* **Sensibilità ROI**: alza/abbassa soglie di `conf` e `area`, o modifica il kernel di dilatazione.
* **Limite deskew**: amplia/limita il range `±20°`.
* **VLM**: aggiorna `MODEL_ID`, disattiva quantizzazione rimuovendo `BitsAndBytesConfig` (richiede più VRAM), o perfeziona il **prompt**.

---

## Prestazioni & consigli

* **GPU**: fortemente consigliata per il VLM; `load_in_8bit=True` riduce VRAM.
* **Batch**: lo script carica modello/tokenizer **una sola volta**; le immagini vengono iterate in sequenza.
* **ROI prima dell’OCR**: riduce rumore e contesto non pertinente, migliorando stabilità della trascrizione.

---

## Risoluzione problemi

* **`Nessuna immagine in input`**: verifica la cartella `input/` e le estensioni.
* **CUDA non disponibile / lento**: controlla installazione PyTorch con CUDA o forza `DEVICE="cpu"` (lento ma funziona).
* **Out Of Memory (VRAM)**: chiudi altre app GPU, mantieni `load_in_8bit=True`, usa immagini più piccole o cambia modello.
* **Riconoscimento povero**: amplia `ALLOWLIST`, aggiungi lingua a EasyOCR, o riduci aggressività del crop (kernel più piccolo).

---

### Note licenze

* Modelli e pesi da Hugging Face seguono le rispettive licenze.
* Librerie: PyTorch, torchvision, OpenCV, Pillow, NumPy, Transformers, EasyOCR.

> Suggerimento: se la prima esecuzione scarica i pesi, tieni attivo l'ambiente e non interrompere.