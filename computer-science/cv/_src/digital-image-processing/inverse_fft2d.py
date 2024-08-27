# Created by modifying "Two-Dimensional Fourier Image Reconstruction (Inverse FT) Demo using Matlab" (© Kota S. Sasaki and Izumi Ohzawa (Licensed under BSD))
# https://visiome.neuroinf.jp/database/item/6448

# python -i inverse_fft2d.py
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image


def load_image(image_path):
    img = Image.open(image_path).convert("L")  # 画像をグレースケールで読み込み
    img = np.array(img)
    return img


def apply_fourier_component(reconstructed_img, f_transform_shifted, x, y):
    rows, cols = f_transform_shifted.shape
    print(
        f"Selected coordinates: {x=}, {y=}, f_transform_shifted.shape: {rows=}, {cols=}"
    )

    # 座標のバリデーション
    if not (0 <= x < cols and 0 <= y < rows):
        print(
            f"Coordinates out of range: x={x}, y={y}, valid range: x(0 to {cols-1}), y(0 to {rows-1})"
        )
        return reconstructed_img, np.zeros_like(reconstructed_img)

    component = np.zeros_like(f_transform_shifted)
    component[y, x] = f_transform_shifted[y, x]

    print(f"Component magnitude: {np.abs(component[y, x])}")

    component = ifftshift(component)
    sine_wave = np.abs(ifft2(component))

    # 振幅に基づいてスケールを調整
    amplitude = np.abs(f_transform_shifted[y, x])
    if amplitude != 0:
        sine_wave *= amplitude

    print(f"Sine wave max value: {np.max(sine_wave)}")

    reconstructed_img += sine_wave

    print(f"Reconstructed image max value: {np.max(reconstructed_img)}")

    return reconstructed_img, sine_wave


def draw(
    axs,
    reconstructed_img,
    current_sine_wave,
    original_img=None,
    amplitude_spectrum=None,
):
    # オリジナル画像
    if original_img is not None:
        axs[0, 0].imshow(original_img, cmap="gray")
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis("off")

    # 振幅スペクトル
    if amplitude_spectrum is not None:
        axs[0, 1].imshow(np.log(amplitude_spectrum + 1), cmap="gray")
        axs[0, 1].set_title("Amplitude Spectrum")
        axs[0, 1].axis("off")

    # 再構成された画像
    axs[1, 0].imshow(reconstructed_img, cmap="gray")
    axs[1, 0].set_title("Reconstructed Image")
    axs[1, 0].axis("off")

    # 最後に追加されたサイン波
    axs[1, 1].imshow(current_sine_wave, cmap="gray")
    axs[1, 1].set_title("Last Added Sine Wave")
    axs[1, 1].axis("off")


def main():
    args = sys.argv
    print(f"{args=}")
    image_path = args[1]  # 画像のパスを指定
    original_img = load_image(image_path)

    # 2次元フーリエ変換の計算
    f_transform = fft2(original_img)
    f_transform_shifted = fftshift(f_transform)
    amplitude_spectrum = np.abs(f_transform_shifted)  # 振幅スペクトル

    # 再構成された画像の初期化
    reconstructed_img = np.zeros_like(original_img, dtype=np.float64)
    current_sine_wave = np.zeros_like(original_img, dtype=np.float64)

    # インタラクティブモードを有効にする
    plt.ion()

    # 初期表示
    fig, axs = plt.subplots(2, 2)
    plt.tight_layout()
    draw(axs, reconstructed_img, current_sine_wave, original_img, amplitude_spectrum)

    print("Please click on the image to select Fourier components.")

    def on_press(event):
        print(f"onclick, {event=}, {event.xdata=}, {event.ydata=}")
        nonlocal reconstructed_img, current_sine_wave
        x, y = int(event.xdata), int(event.ydata)

        # 選択された成分の取得と画像の再構成
        reconstructed_img, current_sine_wave = apply_fourier_component(
            reconstructed_img, f_transform_shifted, x, y
        )
        draw(
            axs,
            reconstructed_img=reconstructed_img,
            current_sine_wave=current_sine_wave,
        )

    fig.canvas.mpl_connect("motion_notify_event", on_press)
    plt.show()


if __name__ == "__main__":
    main()
