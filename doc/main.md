# main.py — Pipeline OCR con InternVL3 e EasyOCR

## Scopo del file

Questo `main.py` implementa una pipeline OCR robusta per immagini statiche. Gli obiettivi principali sono:

* individuare automaticamente l’orientamento migliore dell’immagine;
* localizzare e ritagliare la regione di interesse (ROI) che contiene il testo con EasyOCR;
* migliorare la leggibilità (enhance + deskew);
* trascrivere il testo con il modello VLM **InternVL3\_5‑8B**;
* salvare per ogni immagine la ROI e la trascrizione in `output/`.

## Architettura ad alto livello

```
[input/*.jpg|png|tiff...]           
        │
        ▼
  read_bgr()  →  best_rotation() ─────────────────────────────┐
        │                                                     │
        ▼                                                     │
  EasyOCR readtext() → mask & morpho → largest component → ROI│
        │                                                     │
        ▼                                                     │
    enhance() → deskew_safe() → best_rotation(ROI)            │
        │                                                     │
        ├── salva ROI (JPEG)                                  │
        ▼                                                     │
  vlm_ocr(InternVL3) → trascrizione (TXT)                     │
        │                                                     │
        └── log: angoli usati / eventuali errori              │
```

## Configurazione chiave

* **MODEL\_ID**: `OpenGVLab/InternVL3_5-8B`
* **IN\_DIR**: cartella sorgente (default `input/`)
* **OUT\_DIR**: cartella di output (default `output/`)
* **DEVICE**: `cuda` se disponibile, altrimenti `cpu`
* **ALLOWLIST**: set di caratteri ammessi per OCR
* **ROT\_ANGLES**: angoli testati per auto‑orientamento `[0, 90, 180, 270]`

## Flusso della pipeline

1. **Listing file**: raccoglie immagini supportate in `IN_DIR`.
2. **Bootstrap modelli**: istanzia `easyocr.Reader` e carica InternVL3 (8‑bit quantization).
3. **Per ogni immagine**:

   * Correzione orientamento globale via `best_rotation()` (metrica `ocr_score`).
   * Rilevamento testo con EasyOCR; costruzione di una **mask**; dilatazione morfologica; selezione del **connected component** più grande per definire la ROI.
   * Migliorie visive: `enhance()` (CLAHE + upscaling) e `deskew_safe()` (raddrizzamento entro ±20°).
   * Scelta orientamento migliore della ROI.
   * OCR VLM con `vlm_ocr()` e salvataggio risultati (ROI `.jpg` + testo `.txt`).

## Funzioni principali (riassunto)

| Funzione                             | Input                                     | Output                       | Scopo                                                                             |
| ------------------------------------ | ----------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------- |
| `read_bgr(path)`                     | `Path`                                    | `np.ndarray (BGR)`           | Legge immagine rispettando EXIF, converte in BGR.                                 |
| `rotate90(bgr, angle)`               | `np.ndarray`, `int` ∈ {0,90,180,270}      | `np.ndarray`                 | Ruota l’immagine di multipli di 90°.                                              |
| `ocr_score(bgr, reader)`             | `np.ndarray`, `easyocr.Reader`            | `float`                      | Valuta una rotazione combinando confidenza, lunghezza testo, area box.            |
| `best_rotation(bgr, reader)`         | `np.ndarray`, `easyocr.Reader`            | `(np.ndarray, int, float)`   | Cerca la rotazione migliore tra 0/90/180/270.                                     |
| `enhance(bgr)`                       | `np.ndarray`                              | `np.ndarray`                 | Migliora contrasto (CLAHE) e fa upscaling ×2.                                     |
| `deskew_safe(bgr)`                   | `np.ndarray`                              | `np.ndarray`                 | Raddrizza testo (fino a ±20°) con trasformazione affine.                          |
| `crop_with_easyocr(path, reader)`    | `Path`, `easyocr.Reader`                  | `(PIL.Image, int, int)`      | Estrae ROI testuale, la migliora, riallinea e restituisce anche gli angoli usati. |
| `load_model()`                       | —                                         | `(AutoModel, AutoTokenizer)` | Carica InternVL3\_5‑8B in 8‑bit.                                                  |
| `vlm_ocr(model, tokenizer, pil_img)` | `AutoModel`, `AutoTokenizer`, `PIL.Image` | `str`                        | Esegue OCR via `model.chat` con prompt rigoroso.                                  |
| `process_folder(in_dir)`             | `Path`                                    | — (side‑effects)             | Applica l’intera pipeline batch e salva file in `OUT_DIR`.                        |

## Dettaglio I/O per funzione

### `read_bgr(path: Path) -> np.ndarray`

* **Input**: `path` percorso immagine.
* **Output**: array BGR `H×W×3` (`uint8`).
* **Note**: usa `ImageOps.exif_transpose` per rispettare l’orientamento EXIF.

### `rotate90(bgr: np.ndarray, angle: int) -> np.ndarray`

* **Input**: immagine BGR; `angle ∈ {0,90,180,270}`.
* **Output**: immagine ruotata.

### `ocr_score(bgr: np.ndarray, reader: easyocr.Reader) -> float`

* **Input**: immagine BGR; reader EasyOCR già istanziato.
* **Output**: punteggio (maggiore è meglio).
* **Heuristica**: `Σ conf * (len(text)+1) * log10(10+area)` per ogni box.

### `best_rotation(bgr, reader) -> (best_img, best_a, best_s)`

* **Input**: BGR, reader.
* **Output**: immagine ruotata migliore, angolo, score.

### `enhance(bgr) -> bgr`

* **Input**: BGR.
* **Output**: BGR migliorata (CLAHE canale Y + resize ×2).

### `deskew_safe(bgr) -> bgr`

* **Input**: BGR.
* **Output**: BGR raddrizzata. Limite ±20° per stabilità.

### `crop_with_easyocr(path, reader) -> (roi_pil, ang_used, ang_roi)`

* **Input**: `path`, `reader`.
* **Output**: `roi_pil` (PIL), `ang_used` (int), `ang_roi` (int).
* **Pipeline interna**: best\_rotation(img) → EasyOCR boxes → mask+dilate → largest CC → crop → enhance → deskew → best\_rotation(ROI).

### `load_model() -> (model, tok)`

* **Output**: modello e tokenizer di InternVL3\_5‑8B in 8‑bit (BitsAndBytes).
* **Nota**: `device_map="auto"`, `dtype=float16`, `do_sample=False` usato poi in `vlm_ocr`.

### `vlm_ocr(model, tokenizer, pil_img) -> str`

* **Input**: modello, tokenizer, ROI PIL.
* **Output**: testo OCR.
* **Prompt**: OCR "rigoroso" (caratteri ammessi, nessun formato extra, linee top‑to‑bottom).

### `process_folder(in_dir: Path)`

* **Input**: `in_dir` con immagini (`jpg/jpeg/png/bmp/webp/tif/tiff`).
* **Side‑effects / Output**: in `OUT_DIR` crea per ciascuna immagine:

  * `<stem>_roi.jpg` (ROI salvata)
  * `<stem>_ocr.txt` (trascrizione)
  * log a console con angoli di rotazione.

## Dipendenze principali

* `opencv-python` (cv2)
* `easyocr`
* `torch`, `transformers`
* `Pillow`, `torchvision`, `numpy`

## Esecuzione

```bash
python main.py
```

* Immagini di input in `input/`.
* Risultati in `output/`.

## Note operative

* **GPU** consigliata: il caricamento di InternVL3\_5‑8B in 8‑bit riduce la RAM/VRAM ma resta impegnativo.
* Il **deskew** è limitato a ±20° per evitare correzioni eccessive su immagini rumorose.
* L’**allowlist** privilegia maiuscole e simboli comuni a schemi tecnici/seriali. Adatta se necessario.
