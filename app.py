from pathlib import Path
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from train import build_model
from gradcam import GradCAM, get_target_layer

st.set_page_config(page_title="NeuroVision", page_icon="🧠", layout="wide")
st.title("🧠 NeuroVision")
st.caption("Explainable brain MRI classification — research and educational prototype")

CHECKPOINTS = {
    "EfficientNetB0": Path("models/efficientnet_b0.pth"),
    "ResNet50": Path("models/resnet50.pth"),
    "Custom CNN": Path("models/custom_cnn.pth"),
}
st.warning("Not a medical diagnostic tool. Do not use predictions for clinical decisions.")

with st.sidebar:
    st.header("Analysis settings")
    selected = st.selectbox("Model checkpoint", list(CHECKPOINTS.keys()))
    st.write("Upload a brain MRI image to inspect a trained model's output.")
    st.caption("The selected model must be trained first using train.py.")

uploaded = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg", "bmp"])
checkpoint_path = CHECKPOINTS[selected]

if not checkpoint_path.exists():
    st.info(f"No checkpoint found at `{checkpoint_path}`. Train this model first; see README.md.")
    st.code(f"python train.py --data-dir data --model { {'EfficientNetB0':'efficientnet_b0','ResNet50':'resnet50','Custom CNN':'custom_cnn'}[selected] } --epochs 3", language="bash")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    if checkpoint_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device)
        classes = ckpt["classes"]
        size = ckpt.get("image_size", 224)
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(image).unsqueeze(0).to(device)
        model = build_model(ckpt["model_name"], len(classes)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with torch.no_grad():
            logits = model(tensor)
            scores = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(scores))
        with right:
            st.subheader("Model output")
            st.metric("Predicted dataset class", classes[idx])
            st.caption(f"Highest model score: {scores[idx]*100:.1f}% (not a calibrated clinical probability)")
            st.bar_chart({name: float(score) for name, score in zip(classes, scores)})

        if ckpt["model_name"] in ("resnet50", "efficientnet_b0"):
            st.subheader("Grad-CAM visualization")
            st.caption("Approximate model sensitivity visualization; not a clinical explanation.")
            cam_helper = GradCAM(model, get_target_layer(model, ckpt["model_name"]))
            try:
                cam, _ = cam_helper(tensor, idx)
                resized = image.resize((size, size))
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.imshow(resized)
                ax.imshow(cam, cmap="jet", alpha=0.38)
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            finally:
                cam_helper.close()

        report = "NeuroVision — Research Prototype\n"
        report += f"Image: {uploaded.name}\nModel: {ckpt['model_name']}\n"
        report += f"Predicted dataset class: {classes[idx]}\n"
        report += "Model scores:\n" + "\n".join(f"- {c}: {s:.4f}" for c, s in zip(classes, scores))
        report += "\n\nNot a medical diagnostic tool."
        st.download_button("Download text report", report, file_name="neurovision_report.txt", mime="text/plain")

st.divider()
st.caption("For hackathon demonstration only. Dataset performance does not establish real-world clinical validity.")
