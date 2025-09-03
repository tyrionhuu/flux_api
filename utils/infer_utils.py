import loguru
from PIL import Image

# Same list the community and diffusers issues reference for Kontext
logger = loguru.logger

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]  # https://github.com/huggingface/diffusers/issues/11942

# Extend with halved resolutions for more flexibility
_halved = []
for w, h in PREFERRED_KONTEXT_RESOLUTIONS:
    hw, hh = max(1, w // 2), max(1, h // 2)
    _halved.append((hw, hh))

# Deduplicate while preserving order: original list first, then unique halves
_seen = set(PREFERRED_KONTEXT_RESOLUTIONS)
EXTENDED_KONTEXT_RESOLUTIONS = list(PREFERRED_KONTEXT_RESOLUTIONS)
for wh in _halved:
    if wh not in _seen:
        EXTENDED_KONTEXT_RESOLUTIONS.append(wh)
        _seen.add(wh)


def nearest_kontext_size(w: int, h: int) -> tuple[int, int]:
    # Prefer candidates that do not exceed the current image size (avoid upscaling when possible)
    candidates = [
        (tw, th) for (tw, th) in EXTENDED_KONTEXT_RESOLUTIONS if tw <= w and th <= h
    ]

    def dist2(wh):
        tw, th = wh
        dw, dh = (tw - w), (th - h)
        return dw * dw + dh * dh

    if candidates:
        # Pick the closest by Euclidean distance (squared), tie-break by AR closeness
        ar = w / h if h else 1.0
        return min(candidates, key=lambda wh: (dist2(wh), abs((wh[0] / wh[1]) - ar)))

    # If none fit, fall back to the closest overall (allows upscaling)
    ar = w / h if h else 1.0
    return min(
        EXTENDED_KONTEXT_RESOLUTIONS,
        key=lambda wh: (dist2(wh), abs((wh[0] / wh[1]) - ar)),
    )


def letterbox_to(
    image: Image.Image, target_wh: tuple[int, int], bg=(255, 255, 255)
) -> Image.Image:
    W, H = target_wh
    src_w, src_h = image.width, image.height
    scale = min(W / src_w, H / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (W, H), color=bg)
    # center
    x = (W - new_w) // 2
    y = (H - new_h) // 2
    canvas.paste(resized, (x, y))
    logger.info(f"Resized image to {new_w}x{new_h}")
    return canvas


def kontext_preprocess(image_pil: Image.Image) -> tuple[Image.Image, int, int]:
    # If input image exceeds 512x512 in either dimension, downscale by half first
    src_w, src_h = image_pil.width, image_pil.height
    if src_w > 512 or src_h > 512:
        logger.info(f"Downscaling image from {src_w}x{src_h} to {src_w//2}x{src_h//2}")
        down_w, down_h = max(1, src_w // 2), max(1, src_h // 2)
        image_pil = image_pil.resize((down_w, down_h), Image.Resampling.LANCZOS)

    tgt_w, tgt_h = nearest_kontext_size(image_pil.width, image_pil.height)
    processed = letterbox_to(image_pil, (tgt_w, tgt_h), bg=(255, 255, 255))
    return processed, tgt_w, tgt_h
