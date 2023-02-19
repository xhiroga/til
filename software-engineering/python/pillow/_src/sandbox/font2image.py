from PIL import Image, ImageFont, ImageDraw

# get an image
base = Image.open('lena.png').convert('RGBA')

# make a blank image for the text, initialized to transparent text color
image = Image.new('RGBA', base.size, (255,255,255,0))

draw = ImageDraw.Draw(image)

# use a truetype font
# bitmap fontを使用する場合、load()メソッドを用いてロードする。
font = ImageFont.truetype("FreeMono.ttf", size=128)
draw.text((10, 10), "hello", font=font, fill=(227,72,63,255))
draw.text((60, 138), "world", font=font, fill=(19,122,128,255))

image.save("hello_world.png")