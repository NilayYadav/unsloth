# Real-values probe: what decode_b64_image returns for a phone photo tagged Orientation=6.
import base64, io, os, sys
sys.path.insert(0, os.path.join("studio", "backend"))
from PIL import Image
from core.inference.diffusion import decode_b64_image

RED, GREEN, BLUE, YELLOW = (220, 20, 20), (20, 200, 20), (20, 20, 220), (220, 200, 20)
NAMES = {RED: "red", GREEN: "green", BLUE: "blue", YELLOW: "yellow"}

def photo(w = 64, h = 32, orientation = 6):
    img = Image.new("RGB", (w, h))
    img.paste(RED, (0, 0, w // 2, h // 2)); img.paste(GREEN, (w // 2, 0, w, h // 2))
    img.paste(BLUE, (0, h // 2, w // 2, h)); img.paste(YELLOW, (w // 2, h // 2, w, h))
    ex = img.getexif(); ex[0x0112] = orientation
    buf = io.BytesIO(); img.save(buf, format = "JPEG", quality = 95, subsampling = 0, exif = ex)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def quads(img):
    w, h = img.size
    return [NAMES[min(NAMES, key = lambda c: sum((a - b) ** 2 for a, b in zip(c, img.getpixel((int(w * fx), int(h * fy))))))]
            for fx, fy in ((.25, .25), (.75, .25), (.25, .75), (.75, .75))]

def mask(w, h):
    buf = io.BytesIO(); Image.new("L", (w, h), 255).save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

src = decode_b64_image(photo())
m = decode_b64_image(mask(32, 64), mode = "L")   # canvas is sized from the ORIENTED preview
print("stored JPEG size on the wire ......", Image.open(io.BytesIO(base64.b64decode(photo().partition(',')[2]))).size)
print("browser preview shows ............. (32, 64)")
print("decode_b64_image returns .......... %s" % (src.size,))
print("quadrants TL,TR,BL,BR ............. %s" % (quads(src),))
print("browser-truth quadrants ........... ['blue', 'red', 'yellow', 'green']")
print("inpaint mask size ................. %s" % (m.size,))
print("mask aligns with source ........... %s" % (m.size == src.size))
print("RESULT ............................ %s" % ("PASS" if (src.size == (32, 64) and quads(src) == ["blue", "red", "yellow", "green"] and m.size == src.size) else "FAIL"))
