import html
import json
import re
from pathlib import Path

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

# ──────────────────────────────────────────────────────────────────────────────
# Design system
#   ink #060a11 · panel #0c131f · edge #1b2a3d · signal #5fe3f0 · ember #ff8a3d
#   Sora (display) · IBM Plex Sans (body) · IBM Plex Mono (readouts)
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --ink:#060a11; --panel:#0c131f; --panel2:#101a29; --edge:#1b2a3d; --edge2:#27405c;
  --text:#e8f0fa; --soft:#b3c2d6; --muted:#7d90a8;
  --signal:#5fe3f0; --signal2:#3fa7ff; --ember:#ff8a3d; --ok:#4ade9b;
  --display:'Sora',sans-serif; --body:'IBM Plex Sans',sans-serif; --mono:'IBM Plex Mono',monospace;
}
html,body,[class*="css"]{font-family:var(--body);}
.stApp{
  color:var(--text);
  background:
    radial-gradient(900px 480px at 85% -8%, rgba(63,167,255,.16), transparent 60%),
    radial-gradient(700px 420px at -5% 4%, rgba(95,227,240,.09), transparent 60%),
    linear-gradient(rgba(95,227,240,.035) 1px, transparent 1px) 0 0/44px 44px,
    linear-gradient(90deg, rgba(95,227,240,.035) 1px, transparent 1px) 0 0/44px 44px,
    var(--ink);
}
#MainMenu, footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1360px;padding:1.6rem 2.2rem 3rem;}
h1,h2,h3,h4{font-family:var(--display)!important;letter-spacing:-.03em;color:var(--text);}
p,label,li{color:var(--soft);}
hr{border-color:var(--edge)!important;margin:2.2rem 0!important;}
*:focus-visible{outline:2px solid var(--signal)!important;outline-offset:2px;}

/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#09101a,#070c14);border-right:1px solid var(--edge);}
[data-testid="stSidebar"] .block-container{padding-top:1.4rem;}
.brand{display:flex;align-items:center;gap:13px;margin-bottom:26px;}
.brand-mark{width:44px;height:44px;border-radius:13px;display:grid;place-items:center;font-size:22px;
  background:linear-gradient(145deg,#12384a,#1a2a58);border:1px solid var(--edge2);
  box-shadow:0 0 24px rgba(95,227,240,.22), inset 0 0 12px rgba(95,227,240,.18);}
.brand-name{font:700 19px var(--display);color:var(--text);letter-spacing:-.02em;}
.brand-tag{font:400 12px var(--body);color:var(--muted);}
.side-h{font:600 13px var(--display);color:var(--text);margin:18px 0 6px;}
.pill{display:inline-flex;align-items:center;gap:8px;font:500 12px var(--mono);padding:6px 11px;border-radius:999px;
  border:1px solid var(--edge2);background:rgba(16,26,41,.8);color:var(--soft);}
.pill i{width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 10px var(--ok);}
.pill.off i{background:var(--ember);box-shadow:0 0 10px var(--ember);}
.note{font-size:12px;line-height:1.6;color:var(--muted);border-left:2px solid var(--ember);padding-left:10px;margin-top:6px;}

/* Hero */
.hero-wrap{display:grid;grid-template-columns:1.25fr .75fr;gap:32px;align-items:center;margin:6px 0 26px;}
.hero-title{font:800 clamp(34px,4.4vw,56px)/1.04 var(--display);letter-spacing:-.05em;color:#f4f9ff;margin:0 0 16px;}
.hero-copy{font-size:16px;line-height:1.7;color:var(--soft);max-width:600px;margin:0 0 22px;}
.chips{display:flex;flex-wrap:wrap;gap:9px;}
.chip{font:500 12px var(--mono);color:var(--signal);border:1px solid rgba(95,227,240,.3);background:rgba(95,227,240,.07);padding:6px 11px;border-radius:8px;}
.scanner{position:relative;aspect-ratio:1;max-width:330px;margin:0 auto;}
.scanner svg{width:100%;height:100%;overflow:visible;}
.sweep{transform-origin:150px 150px;animation:sweep 5.5s linear infinite;}
.pulse{animation:pulse 3.2s ease-in-out infinite;transform-origin:150px 150px;}
@keyframes sweep{to{transform:rotate(360deg);}}
@keyframes pulse{0%,100%{opacity:.35}50%{opacity:.9}}
@media (prefers-reduced-motion:reduce){.sweep,.pulse{animation:none;}}
@media (max-width:900px){.hero-wrap{grid-template-columns:1fr;}.scanner{max-width:220px;}}

/* Panels */
.panel{position:relative;background:linear-gradient(160deg,rgba(16,26,41,.92),rgba(10,17,28,.96));
  border:1px solid var(--edge);border-radius:16px;padding:20px 22px;margin-bottom:16px;}
.panel::before{content:"";position:absolute;left:0;top:18px;bottom:18px;width:2px;border-radius:2px;
  background:linear-gradient(var(--signal),var(--signal2));opacity:.8;}
.panel-title{font:700 17px var(--display);color:var(--text);margin-bottom:6px;letter-spacing:-.02em;}
.panel-copy{font-size:13.5px;line-height:1.65;color:var(--muted);}
.steps{display:grid;gap:14px;margin-top:12px;}
.step{display:grid;grid-template-columns:30px 1fr;gap:12px;align-items:start;}
.step b{display:grid;place-items:center;width:30px;height:30px;border-radius:9px;font:600 12px var(--mono);
  color:var(--signal);background:rgba(95,227,240,.08);border:1px solid rgba(95,227,240,.28);}
.step span{font-size:13.5px;color:var(--soft);line-height:1.5;}
.step span em{display:block;font-style:normal;font-weight:600;color:var(--text);}
.sec-h{font:800 28px var(--display);letter-spacing:-.045em;color:var(--text);margin:6px 0 4px;}
.sec-s{color:var(--muted);font-size:14px;margin-bottom:20px;max-width:720px;}

/* Viewer frame (uploaded image) */
[data-testid="stImage"] img{border-radius:12px;border:1px solid var(--edge2);box-shadow:0 0 0 4px rgba(95,227,240,.04),0 20px 50px rgba(0,0,0,.45);}
[data-testid="stImage"] [data-testid="stImageCaption"],[data-testid="stCaptionContainer"]{color:var(--muted)!important;font-family:var(--mono);font-size:12px;}

/* Uploader */
[data-testid="stFileUploader"] section{background:rgba(6,10,17,.6);border:1.5px dashed rgba(95,227,240,.35);border-radius:14px;padding:22px;transition:border-color .2s,background .2s;}
[data-testid="stFileUploader"] section:hover{border-color:var(--signal);background:rgba(95,227,240,.05);}
[data-testid="stFileUploader"] small,[data-testid="stFileUploader"] span{color:var(--muted)!important;}

/* Inputs & buttons */
[data-baseweb="select"]>div{background:var(--panel2)!important;border:1px solid var(--edge2)!important;border-radius:10px!important;}
.stButton>button,.stDownloadButton>button{
  min-height:46px;border-radius:12px;border:1px solid rgba(95,227,240,.5);color:#04141a;font-weight:700;font-family:var(--display);
  background:linear-gradient(100deg,#5fe3f0,#3fa7ff);box-shadow:0 8px 28px rgba(63,167,255,.28);transition:transform .15s,box-shadow .15s;}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-1px);box-shadow:0 12px 34px rgba(63,167,255,.42);color:#04141a;border-color:#9bf2fa;}

/* Verdict */
.verdict{display:grid;grid-template-columns:170px 1fr;gap:26px;align-items:center;
  background:linear-gradient(135deg,rgba(18,40,60,.9),rgba(10,17,28,.96));border:1px solid var(--edge2);border-radius:18px;padding:24px 28px;margin-bottom:18px;
  box-shadow:0 0 60px rgba(63,167,255,.08);}
.ring{position:relative;width:150px;height:150px;}
.ring svg{transform:rotate(-90deg);width:100%;height:100%;}
.ring-val{position:absolute;inset:0;display:grid;place-items:center;text-align:center;font:700 30px var(--display);color:var(--text);letter-spacing:-.04em;}
.ring-val small{display:block;font:400 11px var(--body);color:var(--muted);letter-spacing:0;margin-top:-2px;}
.v-label{font:500 13px var(--body);color:var(--muted);}
.v-class{font:800 clamp(28px,3.4vw,42px)/1.1 var(--display);letter-spacing:-.045em;color:#f5faff;margin:4px 0 14px;overflow-wrap:anywhere;}
.v-meta{display:flex;flex-wrap:wrap;gap:10px;}
.meta{background:rgba(6,10,17,.55);border:1px solid var(--edge);border-radius:10px;padding:8px 13px;}
.meta span{display:block;font:400 11px var(--body);color:var(--muted);}
.meta b{font:600 14px var(--mono);color:var(--text);}
@media (max-width:760px){.verdict{grid-template-columns:1fr;justify-items:center;text-align:center;}.v-meta{justify-content:center;}}

/* Score bars */
.bars{display:grid;gap:13px;margin-top:14px;}
.bar-row .top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;}
.bar-row .name{font:500 14px var(--body);color:var(--soft);}
.bar-row .pct{font:500 13px var(--mono);color:var(--muted);}
.track{height:9px;border-radius:99px;background:rgba(148,163,184,.12);overflow:hidden;}
.fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#2b5f8f,#3b7fb8);}
.bar-row.lead .name{color:var(--text);font-weight:600;}
.bar-row.lead .pct{color:var(--signal);}
.bar-row.lead .fill{background:linear-gradient(90deg,#5fe3f0,#3fa7ff);box-shadow:0 0 14px rgba(95,227,240,.55);}

/* Performance cards */
.perf{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-bottom:16px;}
.perf-card{background:var(--panel);border:1px solid var(--edge);border-radius:16px;padding:20px;}
.perf-card.best{border-color:rgba(95,227,240,.55);box-shadow:0 0 40px rgba(95,227,240,.12);}
.perf-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;}
.perf-name{font:700 16px var(--display);color:var(--text);}
.badge{font:600 11px var(--mono);color:#04141a;background:var(--signal);padding:3px 9px;border-radius:99px;}
.perf-row{margin-bottom:12px;}

/* Streamlit tables */
[data-testid="stDataFrame"]{border:1px solid var(--edge);border-radius:12px;overflow:hidden;}
[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--edge2);background:rgba(16,26,41,.8);}
.footer{text-align:center;color:var(--muted);font-size:12px;line-height:1.8;padding-top:30px;border-top:1px solid var(--edge);margin-top:40px;}
</style>
"""


def ui(markup: str) -> None:
    """Render raw HTML (collapses whitespace so Markdown never treats it as a code block)."""
    st.markdown(re.sub(r"\s*\n\s*", " ", markup.strip()), unsafe_allow_html=True)


def bar_rows(items, lead_first=True):
    out = []
    for i, (name, pct) in enumerate(items):
        cls = "bar-row lead" if (i == 0 and lead_first) else "bar-row"
        out.append(
            f'<div class="{cls}"><div class="top"><span class="name">{html.escape(str(name))}</span>'
            f'<span class="pct">{pct:.1f}%</span></div>'
            f'<div class="track"><div class="fill" style="width:{max(pct, 0.6):.2f}%"></div></div></div>'
        )
    return "".join(out)


def score_ring(pct: float) -> str:
    circ = 2 * np.pi * 62
    offset = circ * (1 - pct / 100)
    return f"""
    <div class="ring">
      <svg viewBox="0 0 150 150">
        <defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#5fe3f0"/><stop offset="1" stop-color="#3fa7ff"/></linearGradient></defs>
        <circle cx="75" cy="75" r="62" fill="none" stroke="rgba(148,163,184,.14)" stroke-width="10"/>
        <circle cx="75" cy="75" r="62" fill="none" stroke="url(#rg)" stroke-width="10" stroke-linecap="round"
                stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"
                style="filter:drop-shadow(0 0 8px rgba(95,227,240,.6))"/>
      </svg>
      <div class="ring-val"><div>{pct:.1f}%<small>model score</small></div></div>
    </div>"""


SCANNER_SVG = """
<div class="scanner"><svg viewBox="0 0 300 300">
  <defs>
    <radialGradient id="core" cx="50%" cy="50%" r="50%">
      <stop offset="0" stop-color="#5fe3f0" stop-opacity=".28"/><stop offset="1" stop-color="#5fe3f0" stop-opacity="0"/></radialGradient>
    <linearGradient id="beam" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#5fe3f0" stop-opacity="0"/><stop offset="1" stop-color="#5fe3f0" stop-opacity=".55"/></linearGradient>
  </defs>
  <circle cx="150" cy="150" r="140" fill="url(#core)"/>
  <circle cx="150" cy="150" r="140" fill="none" stroke="#27405c" stroke-width="1"/>
  <circle cx="150" cy="150" r="105" fill="none" stroke="#1b2a3d" stroke-width="1" stroke-dasharray="3 6"/>
  <circle cx="150" cy="150" r="70" fill="none" stroke="#27405c" stroke-width="1"/>
  <line x1="10" y1="150" x2="290" y2="150" stroke="#1b2a3d"/><line x1="150" y1="10" x2="150" y2="290" stroke="#1b2a3d"/>
  <path class="pulse" d="M150 92c-22-14-52 4-48 34-14 6-18 26-6 38-4 22 18 38 36 30 6 10 24 10 30 0 18 8 40-8 36-30 12-12 8-32-6-38 4-30-26-48-48-34z"
        fill="none" stroke="#5fe3f0" stroke-width="2" stroke-linejoin="round"/>
  <path class="pulse" d="M150 96v92M128 118c10 6 12 16 4 24M172 118c-10 6-12 16-4 24M122 160c12 0 18 6 20 14M178 160c-12 0-18 6-20 14"
        fill="none" stroke="#3fa7ff" stroke-width="1.4" stroke-linecap="round"/>
  <g class="sweep"><path d="M150 150 L290 150 A140 140 0 0 0 250 51 Z" fill="url(#beam)" opacity=".7"/>
    <line x1="150" y1="150" x2="290" y2="150" stroke="#5fe3f0" stroke-width="1.5"/></g>
  <circle cx="214" cy="112" r="5" fill="#ff8a3d"/><circle cx="214" cy="112" r="11" fill="none" stroke="#ff8a3d" stroke-opacity=".6"/>
</svg></div>"""

CHECKPOINTS = {
    "EfficientNetB0": Path("models/efficientnet_b0.pth"),
    "ResNet50": Path("models/resnet50.pth"),
    "Custom CNN": Path("models/custom_cnn.pth"),
}
MODEL_ARGS = {"EfficientNetB0": "efficientnet_b0", "ResNet50": "resnet50", "Custom CNN": "custom_cnn"}

st.markdown(CSS, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    ui('<div class="brand"><div class="brand-mark">🧠</div><div><div class="brand-name">NeuroVision</div>'
       '<div class="brand-tag">Explainable AI lab</div></div></div>')
    ui('<div class="side-h">Model</div>')
    selected = st.selectbox("Model architecture", list(CHECKPOINTS.keys()), label_visibility="collapsed")
    checkpoint_path = CHECKPOINTS[selected]
    if checkpoint_path.exists():
        ui('<span class="pill"><i></i>Checkpoint ready</span>')
    else:
        ui('<span class="pill off"><i></i>Checkpoint missing</span>')
        st.caption(f"Expected at {checkpoint_path.as_posix()}")
    ui('<div class="side-h">Accepted formats</div>')
    ui('<span class="pill">PNG</span> <span class="pill">JPG</span> <span class="pill">JPEG</span> <span class="pill">BMP</span>')
    ui('<div class="side-h">Research use only</div>'
       '<div class="note">Not clinically validated. Do not use for medical decisions.</div>')

# ── Hero ─────────────────────────────────────────────────────────────────────
ui(f"""
<div class="hero-wrap">
  <div>
    <div class="hero-title">Read the scan. See why the model said so.</div>
    <div class="hero-copy">Upload a brain MRI, get class scores from your trained model, and inspect a Grad-CAM map
    showing which regions drove the prediction. Everything runs locally.</div>
    <div class="chips"><span class="chip">Brain MRI classification</span><span class="chip">Class scores</span><span class="chip">Grad-CAM</span></div>
  </div>
  {SCANNER_SVG}
</div>
""")

# ── Input ────────────────────────────────────────────────────────────────────
left, right = st.columns([1.25, .85], gap="large")
with left:
    ui('<div class="panel"><div class="panel-title">Upload a brain MRI</div>'
       '<div class="panel-copy">Choose one image. The selected model processes it on this machine.</div></div>')
    uploaded = st.file_uploader("Choose MRI image", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")
    image_pil = None
    if uploaded is not None:
        image_pil = Image.open(uploaded).convert("RGB")
        st.image(image_pil, caption=f"{uploaded.name}", use_container_width=True)
with right:
    ui("""
    <div class="panel"><div class="panel-title">From scan to explanation</div>
      <div class="steps">
        <div class="step"><b>1</b><span><em>Upload</em>Provide a supported MRI image.</span></div>
        <div class="step"><b>2</b><span><em>Classify</em>Run the selected model checkpoint.</span></div>
        <div class="step"><b>3</b><span><em>Explain</em>Review class scores and the Grad-CAM map.</span></div>
      </div>
    </div>
    <div class="panel"><div class="panel-title">Read results with care</div>
      <div class="panel-copy">Scores are model outputs, not calibrated clinical probabilities. Grad-CAM is an approximate
      visualization and does not prove a medically meaningful explanation.</div></div>
    """)

if not checkpoint_path.exists():
    st.info(f"No checkpoint found at `{checkpoint_path.as_posix()}`. You can explore the interface, but predictions need a trained model.")
    st.code(f"python train.py --data-dir data --model {MODEL_ARGS[selected]} --epochs 3", language="powershell")

# ── Results ──────────────────────────────────────────────────────────────────
if uploaded is not None and image_pil is not None and checkpoint_path.exists():
    st.divider()
    ui('<div class="sec-h">Analysis results</div>'
       '<div class="sec-s">A dataset classification result from your model. It is not a medical diagnosis.</div>')
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

    top_pct = float(scores[idx] * 100)
    ui(f"""
    <div class="verdict">
      {score_ring(top_pct)}
      <div>
        <div class="v-label">Predicted dataset class</div>
        <div class="v-class">{html.escape(str(classes[idx]))}</div>
        <div class="v-meta">
          <div class="meta"><span>Model</span><b>{html.escape(str(ckpt['model_name']))}</b></div>
          <div class="meta"><span>Input size</span><b>{size} x {size}</b></div>
          <div class="meta"><span>Device</span><b>{str(device).upper()}</b></div>
        </div>
      </div>
    </div>
    """)

    score_col, cam_col = st.columns([.9, 1.1], gap="large")
    with score_col:
        order = np.argsort(scores)[::-1]
        items = [(classes[i], float(scores[i] * 100)) for i in order]
        ui(f'<div class="panel"><div class="panel-title">Class scores</div>'
           f'<div class="panel-copy">Softmax output for each dataset class, highest first.</div>'
           f'<div class="bars">{bar_rows(items)}</div></div>')
        score_df = pd.DataFrame({"Dataset class": classes, "Model score (%)": (scores * 100).round(2)})
        st.dataframe(score_df.sort_values("Model score (%)", ascending=False), hide_index=True, use_container_width=True)
    with cam_col:
        ui('<div class="panel"><div class="panel-title">Grad-CAM attention map</div>'
           '<div class="panel-copy">Warmer areas had more influence on the predicted class.</div></div>')
        if ckpt["model_name"] in ("resnet50", "efficientnet_b0"):
            cam_helper = GradCAM(model, get_target_layer(model, ckpt["model_name"]))
            try:
                cam, _ = cam_helper(tensor, idx)
                base = image_pil.resize((size, size))
                fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
                fig.patch.set_facecolor("#0c131f")
                axes[0].imshow(base)
                axes[0].set_title("Original", color="#b3c2d6", fontsize=11, pad=8)
                axes[1].imshow(base)
                axes[1].imshow(cam, cmap="inferno", alpha=.5)
                axes[1].set_title("Grad-CAM overlay", color="#5fe3f0", fontsize=11, pad=8)
                for ax in axes:
                    ax.axis("off")
                fig.tight_layout()
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

# ── Model performance ────────────────────────────────────────────────────────
st.divider()
ui('<div class="sec-h">Model performance</div>'
   '<div class="sec-s">Saved evaluation metrics for the models you have trained and evaluated.</div>')
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
    best = df.sort_values("Macro F1", ascending=False).iloc[0]
    cards = []
    for row in comparison:
        is_best = row["Model"] == best["Model"]
        badge = '<span class="badge">Best macro F1</span>' if is_best else ""
        cards.append(
            f'<div class="perf-card{" best" if is_best else ""}"><div class="perf-head">'
            f'<span class="perf-name">{html.escape(row["Model"])}</span>{badge}</div>'
            f'<div class="perf-row">{bar_rows([("Accuracy", row["Accuracy (%)"])], lead_first=is_best)}</div>'
            f'<div class="perf-row">{bar_rows([("Macro F1", row["Macro F1"] * 100)], lead_first=is_best).replace("%</span>", "</span>")}</div>'
            f'</div>'
        )
    ui(f'<div class="perf">{"".join(cards)}</div>')
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.success(f"Best model by saved macro F1: {best['Model']} (Macro F1 = {best['Macro F1']:.4f})")
else:
    ui('<div class="panel"><div class="panel-title">No evaluation metrics yet</div>'
       '<div class="panel-copy">Run evaluation on your trained models. Accuracy and macro F1 will appear here.</div></div>')

ui('<div class="footer">NeuroVision · Explainable brain MRI classification<br>'
   'Educational and research prototype. Not clinically validated. Do not use for medical decisions.</div>')
