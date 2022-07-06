import PIL
from pathlib import Path
from PIL import UnidentifiedImageError
from PIL import Image

path = Path("/home/bishesh/Desktop/DogsVsCats/prepare_data/train").rglob("*.jpg")
for img_p in path:
    try:
        img = Image.open(img_p)
    except PIL.UnidentifiedImageError:
            print(img_p)