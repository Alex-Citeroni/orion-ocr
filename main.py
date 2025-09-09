from pathlib import Path
import cv2, torch, numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# ================== CONFIG ==================
MODEL_ID = "OpenGVLab/InternVL3_5-8B"
IN_DIR = Path("input")  # cartella sorgente
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-/() ,:#"
ROT_ANGLES = [0, 90, 180, 270]  # per auto-orientamento


# ============== UTILS BASE ==============
def read_bgr(path: Path):
    pil = ImageOps.exif_transpose(Image.open(path))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def rotate90(bgr, angle):
    if angle == 0:
        return bgr
    if angle == 90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


# punteggio OCR per scegliere la rotazione migliore (più box & conf & lunghezza)
def ocr_score(bgr, reader):
    det = reader.readtext(bgr, detail=1, paragraph=False, allowlist=ALLOWLIST)
    if not det:
        return 0.0
    score = 0.0
    for box, text, conf in det:
        area = cv2.contourArea(np.array(box, np.int32))
        score += max(conf, 0) * (len(text) + 1) * np.log10(10 + area)
    return score


def best_rotation(bgr, reader):
    best_a, best_s, best_img = 0, float("-inf"), bgr
    for a in ROT_ANGLES:
        img = rotate90(bgr, a)
        s = ocr_score(img, reader)
        if s > best_s:
            best_s, best_a, best_img = s, a, img
    return best_img, best_a, best_s


def enhance(bgr):
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(
        yuv[:, :, 0]
    )
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return cv2.resize(bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


def deskew_safe(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bitwise_not(g)
    g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(g > 0))
    if len(coords) < 50:
        return bgr
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    # limita a ±20° per evitare giravolte
    angle = float(np.clip(angle, -20, 20))
    (h, w) = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


# ============== ROI CON EASYOCR ==============
def crop_with_easyocr(path, reader):
    bgr0 = read_bgr(path)

    # 1) scegli orientamento migliore PRIMA del crop
    bgr, ang_used, _ = best_rotation(bgr0, reader)

    H, W = bgr.shape[:2]
    det = reader.readtext(bgr, detail=1, paragraph=False, allowlist=ALLOWLIST)

    mask = np.zeros((H, W), np.uint8)
    heights = []
    boxes = []
    for r in det:
        box, text = r[0], r[1]
        conf = r[2] if len(r) > 2 else 1.0
        poly = np.array(box).astype(int)
        wbox = max(np.linalg.norm(poly[1] - poly[0]), np.linalg.norm(poly[2] - poly[3]))
        hbox = max(np.linalg.norm(poly[3] - poly[0]), np.linalg.norm(poly[2] - poly[1]))
        area = wbox * hbox
        if conf >= 0.3 and area > 150:
            cv2.fillPoly(mask, [poly], 255)
            boxes.append(poly)
            heights.append(hbox)

    if not boxes:
        roi = bgr
    else:
        med_h = np.median(heights)
        kx = int(max(25, 3.5 * med_h))
        ky = int(max(8, 1.5 * med_h))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        mask_d = cv2.dilate(mask, kernel, iterations=1)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_d, connectivity=8)
        best_id = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        x, y, w, h, _ = stats[best_id]
        pad = int(0.12 * max(w, h))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        roi = bgr[y0:y1, x0:x1]

    # 2) enhance + deskew
    roi = deskew_safe(enhance(roi))

    # 3) scegli orientamento migliore SULLA ROI
    roi_best, ang_roi, _ = best_rotation(roi, reader)

    return Image.fromarray(cv2.cvtColor(roi_best, cv2.COLOR_BGR2RGB)), ang_used, ang_roi


# ============== MODEL ==============
def load_model():
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False
    )
    return model, tok


# preprocess immagine singola per InternVL3 (no tiling)
TFM = T.Compose(
    [
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def vlm_ocr(model, tokenizer, pil_img: Image.Image) -> str:
    pixel_values = TFM(pil_img).unsqueeze(0).to(dtype=torch.float16, device=DEVICE)
    prompt = (
        "<image>\nYou are a strict OCR engine. Transcribe EXACTLY the printed text, "
        "line by line top-to-bottom. Preserve dashes/slashes/punctuation. "
        "Allowed chars: [A-Z 0-9 .-/:,# ]. If unsure, use '?'. Plain text only."
    )
    gen_cfg = dict(max_new_tokens=256, do_sample=False)
    with torch.inference_mode():
        return model.chat(tokenizer, pixel_values, prompt, gen_cfg)


# ============== BATCH DRIVER ==============
def process_folder(in_dir: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff")
    files = sorted([p for ext in exts for p in in_dir.glob(ext)])
    assert files, f"Nessuna immagine in {in_dir.resolve()}"

    # init una volta
    from easyocr import Reader

    reader = Reader(["en"], gpu=torch.cuda.is_available())

    model, tokenizer = load_model()

    for i, img_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {img_path.name}")
        try:
            roi_pil, ang0, ang_roi = crop_with_easyocr(img_path, reader)
            roi_file = OUT_DIR / f"{img_path.stem}_roi.jpg"
            roi_pil.save(roi_file)

            raw = vlm_ocr(model, tokenizer, roi_pil)

            (OUT_DIR / f"{img_path.stem}_ocr.txt").write_text(raw, encoding="utf-8")

            print(
                f"  ➤ rot(img)={ang0}°, rot(roi)={ang_roi}°  |  saved: {roi_file.name}"
            )
        except Exception as e:
            print(f"  ✗ errore su {img_path.name}: {e}")


if __name__ == "__main__":
    process_folder(IN_DIR)
    print(f"\nDone. Output in: {OUT_DIR.resolve()}")
