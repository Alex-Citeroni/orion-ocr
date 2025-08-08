# OCR ROI + InternVL3-9B

Ritaglio automatico della regione di testo (ROI), miglioramento immagine e OCR con un VLM (InternVL3-9B) guidato da prompt. Il flusso usa EasyOCR per individuare il testo e scegliere l’orientamento migliore, poi applica CLAHE, upscaling e deskew, quindi passa la ROI al modello vision‑language per la trascrizione.

---

## Caratteristiche

* **Auto‑orientamento** dell’immagine in base a uno **score OCR** (più box, più confidenza, più area → punteggio maggiore).
* **Ritaglio ROI** robusto unendo i box rilevati (dilatazione + componente con area massima).
* **Enhance**: CLAHE sul canale Y, **upscale 2×** e **deskew ±20°**.
* **OCR tramite VLM** (InternVL3‑9B) con **prompt rigido** e **allowlist** di caratteri.
* **Batch** su una cartella di input, salvataggio **ROI** e **testo** per ogni immagine.

## Requisiti

* Python 3.10+
* GPU NVIDIA consigliata (CUDA) per prestazioni; il codice funziona anche su CPU (più lento)

Pacchetti principali:

* `torch`, `torchvision`
* `opencv-python`, `numpy`, `pillow`
* `easyocr`
* `transformers`, `accelerate`, `bitsandbytes`

### requirements.txt (esempio)

```txt
easyocr
opencv-python
pillow
numpy
torch
torchvision
transformers
accelerate
bitsandbytes
```

> **Nota CUDA**: installa la versione di `torch` compatibile con la tua CUDA (vedi sito PyTorch). Se non hai GPU, puoi installare la build CPU.

## Installazione

```bash
# 1) Ambiente
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Dipendenze
pip install -r requirements.txt
```

## Struttura I/O

* **Input**: cartella `input/` con immagini
* **Output**: cartella `output/` con:

  * `nomefile_roi.jpg` – ROI migliorata e deskew
  * `nomefile_ocr.txt` – testo OCR (una trascrizione per immagine)

Formati supportati: `jpg, jpeg, png, bmp, webp, tif, tiff`.

## Configurazione rapida

Nel file Python (sezione **CONFIG**):

* `MODEL_ID`: ID Hugging Face del VLM (`OpenGVLab/InternVL3-9B` di default)
* `IN_DIR`: cartella sorgente (default `input`)
* `OUT_DIR`: cartella di output (default `output`)
* `DEVICE`: auto `cuda`/`cpu`
* `ALLOWLIST`: caratteri consentiti dal prompt OCR
* `ROT_ANGLES`: angoli testati per l’auto‑orientamento (default `0, 90, 180, 270`)

Per EasyOCR, la lingua è impostata su **inglese**: `Reader(["en"], gpu=...)`. Puoi aggiungere altre lingue, ad es. `Reader(["en", "it"])`.

## Esecuzione

Lo script si chiama `main.py` e si trova nella root del progetto:

```bash
python main.py
```

Log tipico:

```
[1/12] IMG_0001.jpg
  ➤ rot(img)=90°, rot(roi)=0°  |  saved: IMG_0001_roi.jpg
...
Done. Output in: /path/to/project/output
```

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

## Personalizzazioni utili

* **Caratteri**: modifica `ALLOWLIST` per includere minuscole o simboli aggiuntivi.
* **Lingue OCR**: cambia `Reader(["en"])`.
* **Sensibilità ROI**: alza/abbassa soglie di `conf` e `area`, o modifica il kernel di dilatazione.
* **Limite deskew**: amplia/limita il range `±20°`.
* **VLM**: aggiorna `MODEL_ID`, disattiva quantizzazione rimuovendo `BitsAndBytesConfig` (richiede più VRAM), o perfeziona il **prompt**.

## Prestazioni & consigli

* **GPU**: fortemente consigliata per il VLM; `load_in_8bit=True` riduce VRAM.
* **Batch**: lo script carica modello/tokenizer **una sola volta**; le immagini vengono iterate in sequenza.
* **ROI prima dell’OCR**: riduce rumore e contesto non pertinente, migliorando stabilità della trascrizione.

## Risoluzione problemi

* **`Nessuna immagine in input`**: verifica la cartella `input/` e le estensioni.
* **CUDA non disponibile / lento**: controlla installazione PyTorch con CUDA o forza `DEVICE="cpu"` (lento ma funziona).
* **Out Of Memory (VRAM)**: chiudi altre app GPU, mantieni `load_in_8bit=True`, usa immagini più piccole o cambia modello.
* **Riconoscimento povero**: amplia `ALLOWLIST`, aggiungi lingua a EasyOCR, o riduci aggressività del crop (kernel più piccolo).

## Licenze & crediti

* Modello VLM: `OpenGVLab/InternVL3-9B` (Hugging Face)
* OCR tradizionale: EasyOCR
* Librerie: PyTorch, torchvision, OpenCV, Pillow, NumPy, Transformers

> Verifica le licenze dei modelli/librerie secondo l’uso previsto.

## Esempio rapido

```bash
# Metti alcune immagini in ./input
python main.py
# Troverai in ./output sia *_roi.jpg che *_ocr.txt
```

---

### FAQ

**Posso usare minuscole o caratteri speciali?**
Sì: modifica `ALLOWLIST` e adegua il prompt.

**Come cambio la lingua dell’OCR tradizionale?**
In `Reader(["en"])` aggiungi i codici lingua (es. `"it"`).

**Serve per forza Internet?**
Solo per scaricare i pesi la prima volta da Hugging Face. L’inferenza è locale.

**Posso usare un altro modello?**
Sì, sostituisci `MODEL_ID` e assicurati che `model.chat(...)` sia supportato o adatta la chiamata di generazione.
