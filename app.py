from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from train import build_model
from gradcam import GradCAM, get_target_layer

st.set_page_config(page_title="NeuroVision | MRI Lab", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--bg:#08111f;--panel:#101d30;--line:rgba(148,163,184,.17);--text:#edf4ff;--muted:#9aabc2;--cyan:#55d6e8}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 8% 0%,rgba(62,108,185,.18),transparent 38%),radial-gradient(ellipse at 90% 5%,rgba(57,203,220,.09),transparent 32%),var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(8,17,31,.8)}
[data-testid="stSidebar"]{background:#0b1626;border-right:1px solid var(--line)}
.block-container{max-width:1400px;padding-top:2rem;padding-bottom:3rem}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;color:var(--text)}
p,label{color:#d4deec}
.brand{display:flex;gap:12px;align-items:center;margin-bottom:28px}
.brand-icon{font-size:26px;background:linear-gradient(135deg,#163e56,#27376c);border:1px solid #31657c;border-radius:14px;padding:10px}
.brand-name{font:800 21px Manrope,sans-serif;color:#f4f8ff}
.muted{font-size:12px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase}
.eyebrow{color:var(--cyan);font-size:11px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;margin-bottom:12px}
.hero{font:800 clamp(34px,4vw,50px)/1.08 Manrope,sans-serif;letter-spacing:-.055em;color:#f5f8ff;margin:0 0 14px}
.hero span{background:linear-gradient(90deg,#61dbe8,#91a7ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.intro{color:#a9bad0;line-height:1.7;max-width:760px;font-size:15px;margin-bottom:20px}
.panel{background:linear-gradient(145deg,rgba(19,34,56,.95),rgba(13,25,42,.96));border:1px solid var(--line);border-radius:18px;padding:21px;margin-bottom:16px;box-shadow:0 18px 50px rgba(0,0,0,.12)}
.kicker{color:#83dce8;font-size:11px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;margin-bottom:9px}
.panel-title{color:#f0f5ff;font:750 19px Manrope,sans-serif;margin-bottom:7px}
.panel-copy{color:#9eb0c8;font-size:13px;line-height:1.65}
.metric-card{background:rgba(16,28,45,.9);border:1px solid var(--line);border-radius:14px;padding:17px;min-height:96px}
.metric-label{color:#91a5bf;font-size:11px;text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin-bottom:8px}
.metric-value{color:#f3f7ff;font:800 22px Manrope,sans-serif;overflow-wrap:anywhere}
.section-heading{font:800 26px Manrope,sans-serif;letter-spacing:-.04em;color:#f0f5ff;margin:12px 0 5px}
.section-sub{color:#94a9c3;font-size:13px;margin-bottom:20px}
.stButton>button,.stDownloadButton>button{border-radius:11px;border:1px solid #367d91;background:linear-gradient(100deg,#1b91a8,#4e6fd6);color:white;font-weight:700;min-height:42px}
[data-testid="stFileUploader"]{background:rgba(8,17,31,.45);border:1px dashed rgba(108,190,215,.38);border-radius:14px;padding:10px}
[data-testid="stMetric"]{background:rgba(16,28,45,.8);border:1px solid var(--line);padding:14px;border-radius:14px}
hr{border-color:var(--line)}
.footer{text-align:center;color:#70839c;font-size:11px;line-height:1.8;padding-top:25px}
</style>
""", unsafe_allow_html=True)

CHECKPOINTS = {
    "EfficientNetB0": Path("models/efficientnet_b0.pth"),
    "ResNet50": Path("models/resnet50.pth"),
    "Custom CNN": Path("models/custom_cnn.pth"),
}
MODEL_ARGS = {"EfficientNetB0": "efficientnet_b0", "ResNet50": "resnet50", "Custom CNN": "custom_cnn"}

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">🧠</div><div><div class="brand-name">NeuroVision</div><div class="muted">Explainable AI Lab</div></div></div>', unsafe_allow_html=True)
    st.markdown("### Analysis workspace")
    st.caption("Choose a trained model and upload one MRI image.")
    selected = st.selectbox("Model architecture", list(CHECKPOINTS.keys()))
    checkpoint_path = CHECKPOINTS[selected]
    if checkpoint_path.exists():
        st.success("Checkpoint detected")
    else:
        st.warning("Checkpoint not found")
        st.caption(f"Expected: {checkpoint_path.as_posix()}")
    st.divider()
    st.markdown("**Accepted image formats**")
    st.caption("PNG · JPG · JPEG · BMP")
    st.markdown("**Research use only**")
    st.caption("Not clinically validated. Do not use for medical decisions.")

st.markdown('<div class="eyebrow">NEUROIMAGING · EXPLAINABLE MACHINE LEARNING</div><div class="hero">Understand the image.<br><span>Explore the model.</span></div><div class="intro">An interactive workspace for brain MRI classification, model-score inspection, and Grad-CAM visualization. Upload a scan to explore how a trained model responds.</div>', unsafe_allow_html=True)

left, right = st.columns([1.25, .85], gap="large")
with left:
    st.markdown('<div class="panel"><div class="kicker">01 · Image input</div><div class="panel-title">Upload a brain MRI</div><div class="panel-copy">Choose a single supported MRI image. The selected trained model will process it locally.</div></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose MRI image", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")
    image_pil = None
    if uploaded is not None:
        image_pil = Image.open(uploaded).convert("RGB")
        st.image(image_pil, caption=f"Selected image · {uploaded.name}", use_container_width=True)
with right:
    st.markdown('<div class="panel"><div class="kicker">Workflow overview</div><div class="panel-title">From scan to explanation</div><div class="panel-copy"><b>01 · Upload</b><br>Provide a supported MRI image.<br><br><b>02 · Classify</b><br>Run the selected model checkpoint.<br><br><b>03 · Explain</b><br>Inspect class scores and Grad-CAM when supported.</div></div><div class="panel"><div class="kicker">Important note</div><div class="panel-title">Research use only</div><div class="panel-copy">Scores are model outputs, not calibrated clinical probabilities. Grad-CAM is an approximate visualization, not proof of a medically meaningful explanation.</div></div>', unsafe_allow_html=True)

if not checkpoint_path.exists():
    st.info(f"No checkpoint found at `{checkpoint_path.as_posix()}`. The interface can be viewed, but predictions require a trained model.")
    st.code(f"python train.py --data-dir data --model {MODEL_ARGS[selected]} --epochs 3", language="powershell")

if uploaded is not None and image_pil is not None and checkpoint_path.exists():
    st.divider()
    st.markdown('<div class="section-heading">Analysis results</div><div class="section-sub">Review model outputs below. This is a dataset classification result, not a medical diagnosis.</div>', unsafe_allow_html=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.spinner("Loading model and analyzing image..."):
        ckpt = torch.load(checkpoint_path, map_location=device)
        classes = ckpt["classes"]
        size = ckpt.get("image_size", 224)
        transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(image_pil).unsqueeze(0).to(device)
        model = build_model(ckpt["model_name"], len(classes)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            scores = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(scores))

    m1, m2, m3 = st.columns(3)
    for col, label, value in [
        (m1, "Predicted dataset class", str(classes[idx])),
        (m2, "Highest model score", f"{scores[idx] * 100:.1f}%"),
        (m3, "Inference device", str(device).upper()),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

    score_col, cam_col = st.columns([.95, 1.05], gap="large")
    with score_col:
        st.markdown('<div class="kicker">02 · Classification</div><div class="panel-title">Class score breakdown</div>', unsafe_allow_html=True)
        score_df = pd.DataFrame({"Dataset class": classes, "Model score (%)": (scores * 100).round(2)})
        st.bar_chart(score_df.set_index("Dataset class"), horizontal=True)
        st.dataframe(score_df.sort_values("Model score (%)", ascending=False), hide_index=True, use_container_width=True)
    with cam_col:
        st.markdown('<div class="kicker">03 · Explainability</div><div class="panel-title">Grad-CAM attention map</div>', unsafe_allow_html=True)
        if ckpt["model_name"] in ("resnet50", "efficientnet_b0"):
            cam_helper = GradCAM(model, get_target_layer(model, ckpt["model_name"]))
            try:
                cam, _ = cam_helper(tensor, idx)
                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor("#101d30")
                ax.imshow(image_pil.resize((size, size)))
                ax.imshow(cam, cmap="jet", alpha=.4)
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            finally:
                cam_helper.close()
        else:
            st.info("Grad-CAM is available for ResNet50 and EfficientNetB0.")
    report = f"NeuroVision — Research Prototype\nImage: {uploaded.name}\nModel: {ckpt['model_name']}\nPredicted dataset class: {classes[idx]}\nModel scores:\n"
    report += "\n".join(f"- {c}: {s:.4f}" for c, s in zip(classes, scores))
    report += "\n\nNot a medical diagnostic tool."
    st.download_button("Download analysis report", report, file_name="neurovision_report.txt", mime="text/plain", use_container_width=True)

st.divider()
st.markdown('<div class="section-heading">Model performance</div><div class="section-sub">Compare saved evaluation metrics from models you have trained and evaluated.</div>', unsafe_allow_html=True)
metric_files = {
    "EfficientNetB0": "outputs/efficientnet_b0_metrics.json",
    "ResNet50": "outputs/resnet50_metrics.json",
    "Custom CNN": "outputs/custom_cnn_metrics.json",
}
comparison = []
for model_name, file_path in metric_files.items():
    path = Path(file_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            comparison.append({"Model": model_name, "Accuracy (%)": round(metrics["accuracy"] * 100, 2), "Macro F1": round(metrics["macro_f1"], 4)})
        except (OSError, ValueError, KeyError, TypeError):
            pass
if comparison:
    df = pd.DataFrame(comparison)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="panel-title">Accuracy</div>', unsafe_allow_html=True)
        st.bar_chart(df.set_index("Model")[["Accuracy (%)"]])
    with c2:
        st.markdown('<div class="panel-title">Macro F1-score</div>', unsafe_allow_html=True)
        st.bar_chart(df.set_index("Model")[["Macro F1"]])
    st.dataframe(df, hide_index=True, use_container_width=True)
    best = df.sort_values("Macro F1", ascending=False).iloc[0]
    st.success(f"Best model by saved macro F1: {best['Model']} (Macro F1 = {best['Macro F1']:.4f})")
else:
    st.markdown('<div class="panel"><div class="panel-title">Evaluation metrics will appear here</div><div class="panel-copy">Run evaluation for trained models to display accuracy and macro F1-score here.</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">NEUROVISION · EXPLAINABLE BRAIN MRI CLASSIFICATION<br>Educational and research prototype only · Not clinically validated · Do not use for medical decisions</div>', unsafe_allow_html=True)
