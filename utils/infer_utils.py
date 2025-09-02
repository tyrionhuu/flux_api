from PIL import Image

# Same list the community and diffusers issues reference for Kontext
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672,1568),(688,1504),(720,1456),(752,1392),(800,1328),(832,1248),
    (880,1184),(944,1104),(1024,1024),(1104,944),(1184,880),(1248,832),
    (1328,800),(1392,752),(1456,720),(1504,688),(1568,672),
]  # https://github.com/huggingface/diffusers/issues/11942


def nearest_kontext_size(w: int, h: int) -> tuple[int, int]:
    ar = w / h if h else 1.0
    return min(PREFERRED_KONTEXT_RESOLUTIONS, key=lambda wh: abs((wh[0] / wh[1]) - ar))


def letterbox_to(image: Image.Image, target_wh: tuple[int, int], bg=(255, 255, 255)) -> Image.Image:
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
    return canvas


def kontext_preprocess(image_pil: Image.Image) -> tuple[Image.Image, int, int]:
    tgt_w, tgt_h = nearest_kontext_size(image_pil.width, image_pil.height)
    processed = letterbox_to(image_pil, (tgt_w, tgt_h), bg=(255, 255, 255))
    return processed, tgt_w, tgt_h


