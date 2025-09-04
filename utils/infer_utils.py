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

# Extend with 2/3 resolutions for more flexibility
_two_thirds = []
for w, h in PREFERRED_KONTEXT_RESOLUTIONS:
    tw, th = max(1, int(w * 2 // 3)), max(1, int(h * 2 // 3))
    _two_thirds.append((tw, th))

# Replace original list with 2/3 resolutions instead of appending
EXTENDED_KONTEXT_RESOLUTIONS = _two_thirds


def nearest_kontext_size(w: int, h: int, use_extended: bool = True) -> tuple[int, int]:
    # Choose resolution list based on downscale setting
    if use_extended:
        resolution_list = EXTENDED_KONTEXT_RESOLUTIONS
        logger.info(f"Using extended resolutions (including 2/3) for {w}x{h} image")
    else:
        resolution_list = PREFERRED_KONTEXT_RESOLUTIONS
        logger.info(f"Using original resolutions (no 2/3) for {w}x{h} image")
    
    # Calculate target aspect ratio
    target_ar = w / h if h else 1.0
    logger.info(f"Target aspect ratio: {target_ar:.3f}")
    
    # Prefer candidates that do not exceed the current image size (avoid upscaling when possible)
    candidates = [
        (tw, th) for (tw, th) in resolution_list if tw <= w and th <= h
    ]

    def aspect_ratio_diff(wh):
        tw, th = wh
        candidate_ar = tw / th if th else 1.0
        return abs(candidate_ar - target_ar)

    if candidates:
        # Pick the closest by aspect ratio difference
        best_candidate = min(candidates, key=aspect_ratio_diff)
        best_ar = best_candidate[0] / best_candidate[1] if best_candidate[1] else 1.0
        logger.info(f"Selected candidate {best_candidate} with AR {best_ar:.3f} (diff: {aspect_ratio_diff(best_candidate):.3f})")
        return best_candidate

    # If none fit, fall back to the closest aspect ratio overall (allows upscaling)
    best_overall = min(resolution_list, key=aspect_ratio_diff)
    best_ar = best_overall[0] / best_overall[1] if best_overall[1] else 1.0
    logger.info(f"Selected overall best {best_overall} with AR {best_ar:.3f} (diff: {aspect_ratio_diff(best_overall):.3f})")
    return best_overall


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


def kontext_preprocess(
    image_pil: Image.Image, downscale: bool = True
) -> tuple[Image.Image, int, int]:
    # If downscale is enabled and input image exceeds 512x512 in either dimension, downscale by 2/3 first
    src_w, src_h = image_pil.width, image_pil.height
    logger.info(f"kontext_preprocess called with downscale={downscale}, image size={src_w}x{src_h}")
    if downscale and (src_w > 512 or src_h > 512):
        logger.info(f"Downscaling image from {src_w}x{src_h} to {int(src_w * 2 // 3)}x{int(src_h * 2 // 3)}")
        down_w, down_h = max(1, int(src_w * 2 // 3)), max(1, int(src_h * 2 // 3))
        image_pil = image_pil.resize((down_w, down_h), Image.Resampling.LANCZOS)
    elif not downscale:
        logger.info(f"Downscaling disabled, keeping original image size {src_w}x{src_h}")

    tgt_w, tgt_h = nearest_kontext_size(image_pil.width, image_pil.height, use_extended=downscale)
    processed = letterbox_to(image_pil, (tgt_w, tgt_h), bg=(255, 255, 255))
    return processed, tgt_w, tgt_h

