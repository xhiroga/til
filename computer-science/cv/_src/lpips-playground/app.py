import lpips
from PIL import Image
import torchvision.transforms as transforms

loss_fn_alex = lpips.LPIPS(net="alex")  # best forward scores
loss_fn_vgg = lpips.LPIPS(
    net="vgg"
)  # closer to "traditional" perceptual loss, when used for optimization

transform = transforms.ToTensor()

img0 = transform(Image.open("data/p0_000000.png").convert("RGB")).unsqueeze(0)
img1 = transform(Image.open("data/p1_000000.png").convert("RGB")).unsqueeze(0)
img2 = transform(Image.open("data/p0_000001.png").convert("RGB")).unsqueeze(0)

print("Similar images:")
d_alex = loss_fn_alex(img0, img1)
d_vgg = loss_fn_vgg(img0, img1)
print(f"{d_vgg=}, {d_alex=}")

print("Different images:")
d_alex = loss_fn_alex(img0, img2)
d_vgg = loss_fn_vgg(img0, img2)
print(f"{d_vgg=}, {d_alex=}")
