from pytesseract import pytesseract
from PIL import Image

# 読み込み対象ファイルの指定
img = Image.open("./receipt_highcontrast_large.jpeg", "r")
# tesseractコマンドのインストールパス
pytesseract.tesseract_cmd = "/usr/bin/tesseract"

boxes = pytesseract.image_to_boxes(img, lang="jpn", output_type=pytesseract.Output.DICT)
print(boxes)
