import gradio as gr


def estimate_hand(image):
    pass

iface = gr.Interface(
    fn=estimate_hand,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Image(type="numpy", label="Hand Estimation Result"),
    title="Hand Pose Estimation",
    description="Upload an image and get hand pose estimation using MediaPipe."
)

if __name__ == "__main__":
    iface.launch()
