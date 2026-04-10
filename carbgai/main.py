# Car AI Background Removal Tool - FastAPI Backend v1.1
#
# SETUP & RUN:
# pip install -r requirements.txt
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Then open index.html in browser (or: python -m http.server 3000)
# First run will download BiRefNet model (~200MB) automatically

import io
import time
import numpy as np
from pathlib import Path

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from PIL import Image, ImageDraw, ImageFilter
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# ─────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────
app = FastAPI(title="CarBG.ai", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
DEVICE = None
TRANSFORM = None
BACKGROUNDS_DIR = Path("backgrounds")

# ─────────────────────────────────────────────
# Background placeholder generation
# ─────────────────────────────────────────────

def generate_gradient(width, height, top_color, bottom_color):
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * t)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * t)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * t)
        for x in range(width):
            pixels[x, y] = (r, g, b)
    return img

def generate_showroom(width, height):
    img = Image.new("RGB", (width, height), (10, 10, 26))
    draw = ImageDraw.Draw(img)
    floor_y = int(height * 0.65)
    for y in range(floor_y, height):
        t = (y - floor_y) / max(height - floor_y, 1)
        draw.line([(0, y), (width, y)], fill=(int(10+20*t), int(10+20*t), int(26+30*t)))
    for i in range(40):
        draw.line([(0, floor_y - i), (width, floor_y - i)], fill=(30+i, 30+i, 80+i))
    return img

def generate_outdoor(width, height):
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    horizon = int(height * 0.55)
    for y in range(horizon):
        t = y / max(horizon, 1)
        pixels_row = (int(100+(80*t)), int(160+(50*t)), int(220+(15*t)))
        for x in range(width): pixels[x, y] = pixels_row
    for y in range(horizon, height):
        t = (y - horizon) / max(height - horizon, 1)
        pixels_row = (int(130-(30*t)), int(145-(30*t)), int(110-(25*t)))
        for x in range(width): pixels[x, y] = pixels_row
    return img

def ensure_backgrounds():
    BACKGROUNDS_DIR.mkdir(exist_ok=True)
    W, H = 1280, 720
    files = {
        "studio_white.jpg": lambda: generate_gradient(W, H, (245,245,245), (224,224,224)),
        "studio_grey.jpg":  lambda: generate_gradient(W, H, (42,42,42), (64,64,64)),
        "showroom.jpg":     lambda: generate_showroom(W, H),
        "outdoor.jpg":      lambda: generate_outdoor(W, H),
    }
    for fname, gen in files.items():
        fpath = BACKGROUNDS_DIR / fname
        if not fpath.exists():
            print(f"  Generating: {fname}")
            gen().save(str(fpath), "JPEG", quality=92)

# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global MODEL, DEVICE, TRANSFORM
    print("\n" + "═"*50)
    print("  CarBG.ai v1.1 — Starting up")
    print("═"*50)

    print("\n[1/2] Ensuring background images...")
    ensure_backgrounds()
    print("  ✓ Backgrounds ready")

    print("\n[2/2] Loading BiRefNet model (first run ~200MB download)...")
    t0 = time.time()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {DEVICE}")
    MODEL = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    MODEL.to(DEVICE)
    MODEL.eval()
    TRANSFORM = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"  ✓ Model ready in {time.time()-t0:.1f}s")
    print("\n  → http://localhost:8000  |  open index.html in browser\n")

# ─────────────────────────────────────────────
# Processing helpers
# ─────────────────────────────────────────────

def run_birefnet(image: Image.Image) -> Image.Image:
    orig_size = image.size
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = MODEL(tensor)
    pred = output[-1].sigmoid().squeeze()
    mask_np = (pred.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(mask_np, mode="L").resize(orig_size, Image.LANCZOS)

def create_cutout(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba

def smart_resize_car(car: Image.Image, bg_width: int, max_frac: float = 0.75) -> Image.Image:
    max_w = int(bg_width * max_frac)
    w, h = car.size
    if w > max_w:
        car = car.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    return car

def add_drop_shadow(bg: Image.Image, cx, cy, car_w, car_h) -> Image.Image:
    layer = Image.new("RGBA", bg.size, (0,0,0,0))
    draw = ImageDraw.Draw(layer)
    ew, eh = int(car_w*0.80), int(car_h*0.10)
    draw.ellipse([cx-ew//2, cy+car_h-eh//2, cx+ew//2, cy+car_h+eh//2], fill=(0,0,0,70))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=20))
    return Image.alpha_composite(bg.convert("RGBA"), layer)

def add_reflection(bg: Image.Image, car: Image.Image, car_x, car_y) -> Image.Image:
    car_w, car_h = car.size
    flipped = car.transpose(Image.FLIP_TOP_BOTTOM)
    crop_h = int(car_h * 0.30)
    refl = flipped.crop((0, 0, car_w, crop_h))
    refl_np = np.array(refl).astype(np.float32)
    if refl_np.shape[2] < 4:
        refl_np = np.dstack([refl_np, np.ones((crop_h, car_w), np.float32)*255])
    for row in range(crop_h):
        refl_np[row, :, 3] *= 0.35 * (1.0 - row / max(crop_h-1, 1))
    refl = Image.fromarray(np.clip(refl_np, 0, 255).astype(np.uint8), "RGBA")
    bg_rgba = bg.convert("RGBA")
    ry = car_y + car_h
    bh = bg_rgba.size[1]
    if ry < bh:
        if ry + crop_h > bh:
            refl = refl.crop((0, 0, car_w, bh - ry))
        bg_rgba.paste(refl, (car_x, ry), refl)
    return bg_rgba

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    bg_choice: str = Form(default="studio_white"),
    custom_bg: UploadFile = File(default=None),
):
    """
    Process car image with background removal.
    - file        : car image (required)
    - bg_choice   : preset name OR "custom" when custom_bg is supplied
    - custom_bg   : optional user-uploaded background image
    """
    try:
        # Load & segment car
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        mask = run_birefnet(image)
        car_cutout = create_cutout(image, mask)

        # Load background — custom or preset
        if bg_choice == "custom" and custom_bg is not None:
            bg_bytes = await custom_bg.read()
            background = Image.open(io.BytesIO(bg_bytes)).convert("RGB")
        else:
            bg_path = BACKGROUNDS_DIR / f"{bg_choice}.jpg"
            if not bg_path.exists():
                bg_path = BACKGROUNDS_DIR / "studio_white.jpg"
            background = Image.open(str(bg_path)).convert("RGB")

        background = background.resize((1280, 720), Image.LANCZOS)
        bg_w, bg_h = background.size

        # Resize car, compute position
        car_cutout = smart_resize_car(car_cutout, bg_w, 0.75)
        car_w, car_h = car_cutout.size
        car_x = (bg_w - car_w) // 2
        car_y = int(bg_h * 0.78) - car_h

        # Floor gradient darkening
        floor_layer = Image.new("RGBA", (bg_w, bg_h), (0,0,0,0))
        fd = ImageDraw.Draw(floor_layer)
        fys = int(bg_h * 0.70)
        for y in range(fys, bg_h):
            t = (y - fys) / max(bg_h - fys, 1)
            fd.line([(0, y), (bg_w, y)], fill=(0, 0, 0, int(40 * t)))
        bg_rgba = Image.alpha_composite(background.convert("RGBA"), floor_layer)

        # Shadow → reflection → car composite
        bg_rgba = add_drop_shadow(bg_rgba, car_x, car_y, car_w, car_h)
        bg_rgba = add_reflection(bg_rgba, car_cutout, car_x, car_y)
        bg_rgba.paste(car_cutout, (car_x, car_y), car_cutout)

        buf = io.BytesIO()
        bg_rgba.convert("RGB").save(buf, format="JPEG", quality=92)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
