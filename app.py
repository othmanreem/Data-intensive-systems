from PIL import Image
import gradio as gr
from controlnet_aux import OpenposeDetector

# Load OpenPose detector
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

def generate_pose(image, use_openpose=True):
    img = image.convert("RGB")
    if use_openpose:
        result = openpose(img)
    else:
        result = img
    if not isinstance(result, Image.Image):
        result = Image.fromarray(result)
    return result

# Gradio UI
demo = gr.Interface(
    fn=generate_pose,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Checkbox(value=True, label="Use OpenPose (default: true)"),
    ],
    outputs=gr.Image(type="pil", label="Pose Output"),
    title="OpenPose Pose Generator",
    description="Generate full body pose including face and hands."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
