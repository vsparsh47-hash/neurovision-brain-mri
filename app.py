import html
import io
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

st.set_page_config(page_title="NeuroVision", page_icon="🧠", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# Design tokens
#   Graphite  #0b0e13 / #10141b / #151a23     Porcelain text  #ece9e2
#   Glacier accent #8ad3c3                     Amber (caution) #e6b372
#   Instrument Serif (display) · Hanken Grotesk (interface) · JetBrains Mono (readouts)
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg:#0b0e13; --s1:#10141b; --s2:#151a23; --s3:#1b212c;
  --line:rgba(255,255,255,.075); --line2:rgba(255,255,255,.14);
  --tx:#ece9e2; --tx2:#b9bcc4; --mu:#858c99;
  --ac:#8ad3c3; --ac-soft:rgba(138,211,195,.12); --warn:#e6b372;
  --serif:'Instrument Serif',Georgia,serif; --sans:'Hanken Grotesk',system-ui,sans-serif; --mono:'JetBrains Mono',monospace;
}
html,body,[class*="css"]{font-family:var(--sans);}
.stApp{color:var(--tx);
  background:radial-gradient(1100px 520px at 78% -12%, rgba(138,211,195,.07), transparent 62%),
             radial-gradient(800px 480px at -8% 8%, rgba(120,140,200,.05), transparent 60%), var(--bg);}
#MainMenu,footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:1240px;padding:1.4rem 2.4rem 3.5rem;}
p,label,li{color:var(--tx2);}
hr{border-color:var(--line)!important;margin:2.8rem 0!important;}
*:focus-visible{outline:2px solid var(--ac)!important;outline-offset:2px;}

/* Top bar */
.top{display:flex;justify-content:space-between;align-items:center;padding:6px 0 22px;border-bottom:1px solid var(--line);margin-bottom:46px;}
.logo{display:flex;align-items:center;gap:11px;font:600 17px var(--sans);letter-spacing:-.01em;color:var(--tx);}
.top-right{display:flex;gap:10px;align-items:center;}
.tag{font:400 12px var(--mono);color:var(--tx2);border:1px solid var(--line2);border-radius:999px;padding:6px 12px;background:var(--s1);}
.tag.live::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--ac);margin-right:8px;vertical-align:1px;box-shadow:0 0 0 3px var(--ac-soft);}
.tag.warn::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--warn);margin-right:8px;vertical-align:1px;}

/* Hero */
.h1{font:400 clamp(42px,5.4vw,76px)/1.0 var(--serif);letter-spacing:-.02em;color:var(--tx);max-width:820px;margin:0 0 20px;}
.lede{font-size:17px;line-height:1.7;color:var(--tx2);max-width:600px;margin:0 0 38px;}

/* Stepper */
.rail{display:flex;gap:0;margin:0 0 44px;max-width:760px;}
.rs{flex:1;position:relative;padding-top:0;}
.rs::before{content:"";position:absolute;top:13px;left:30px;right:6px;height:1px;background:var(--line2);}
.rs:last-child::before{display:none;}
.rs.done::before{background:var(--ac);opacity:.6;}
.dot{width:27px;height:27px;border-radius:50%;display:grid;place-items:center;font:500 12px var(--mono);
  border:1px solid var(--line2);color:var(--mu);background:var(--bg);position:relative;z-index:1;}
.rs.active .dot{border-color:var(--ac);color:var(--ac);box-shadow:0 0 0 4px var(--ac-soft);}
.rs.done .dot{background:var(--ac);border-color:var(--ac);color:#08110f;font-weight:600;}
.rs-t{margin-top:12px;font:600 14px var(--sans);color:var(--tx);}
.rs.idle .rs-t{color:var(--mu);}
.rs-s{font-size:13px;color:var(--mu);margin-top:2px;line-height:1.5;padding-right:18px;}

/* Panels & headings */
.panel{background:var(--s1);border:1px solid var(--line);border-radius:16px;padding:22px 24px;margin-bottom:16px;}
.p-title{font:600 16px var(--sans);color:var(--tx);letter-spacing:-.01em;margin-bottom:4px;}
.p-copy{font-size:14px;line-height:1.65;color:var(--mu);}
.sec{font:400 clamp(30px,3.4vw,42px)/1.08 var(--serif);letter-spacing:-.015em;color:var(--tx);margin:0 0 8px;}
.sec-s{font-size:15px;color:var(--mu);margin-bottom:28px;max-width:640px;line-height:1.6;}

/* Sidebar */
[data-testid="stSidebar"]{background:#090c10;border-right:1px solid var(--line);}
.sb-h{font:600 12px var(--sans);color:var(--mu);margin:22px 0 8px;letter-spacing:.02em;}
.note{font-size:13px;line-height:1.65;color:var(--mu);padding:14px 16px;border:1px solid var(--line);border-radius:12px;background:var(--s1);}
.fmt{display:inline-block;font:400 12px var(--mono);color:var(--tx2);border:1px solid var(--line2);border-radius:7px;padding:3px 9px;margin:0 6px 6px 0;}

/* Upload + image */
[data-testid="stFileUploader"] section{background:var(--s1);border:1px dashed var(--line2);border-radius:16px;padding:30px;transition:border-color .2s,background .2s;}
[data-testid="stFileUploader"] section:hover{border-color:var(--ac);background:var(--s2);}
[data-testid="stImage"] img{border-radius:14px;border:1px solid var(--line2);box-shadow:0 30px 70px rgba(0,0,0,.5);}
[data-testid="stCaptionContainer"],[data-testid="stImageCaption"]{color:var(--mu)!important;font:400 12px var(--mono)!important;}

/* Controls */
[data-baseweb="select"]>div{background:var(--s2)!important;border:1px solid var(--line2)!important;border-radius:10px!important;}
.stButton>button,.stDownloadButton>button{min-height:44px;border-radius:10px;font:600 14px var(--sans);
  background:var(--tx);color:#0b0e13;border:1px solid var(--tx);transition:background .15s,transform .15s;}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--ac);border-color:var(--ac);color:#08110f;transform:translateY(-1px);}
[data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line);}
button[data-baseweb="tab"]{background:transparent;color:var(--mu);font:500 14px var(--sans);padding:12px 16px;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--tx);}
[data-baseweb="tab-highlight"]{background:var(--ac)!important;height:2px!important;}
[data-baseweb="tab-border"]{display:none;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--line2);background:var(--s1);}
[data-testid="stCode"] pre{border-radius:12px;}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{background:var(--ac);}

/* Verdict */
.verdict{display:grid;grid-template-columns:1.35fr .65fr;gap:40px;align-items:end;padding:34px 38px;margin-bottom:18px;
  background:linear-gradient(180deg,var(--s2),var(--s1));border:1px solid var(--line2);border-radius:20px;}
.v-k{font-size:13px;color:var(--mu);margin-bottom:10px;}
.v-class{font:400 clamp(40px,5vw,72px)/1.0 var(--serif);letter-spacing:-.025em;color:var(--tx);overflow-wrap:anywhere;}
.v-num{text-align:right;}
.v-pct{font:400 clamp(48px,5.4vw,84px)/1 var(--serif);letter-spacing:-.03em;color:var(--ac);}
.v-pct small{font-size:.42em;color:var(--mu);margin-left:3px;}
.meter{margin-top:14px;}
.m-track{position:relative;height:5px;border-radius:99px;background:rgba(255,255,255,.09);}
.m-fill{position:absolute;inset:0 auto 0 0;border-radius:99px;background:var(--ac);}
.m-tick{display:flex;justify-content:space-between;font:400 10px var(--mono);color:var(--mu);margin-top:8px;}
.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;}
.fact{border:1px solid var(--line);border-radius:14px;padding:15px 18px;background:var(--s1);}
.fact span{display:block;font-size:12px;color:var(--mu);margin-bottom:5px;}
.fact b{font:500 15px var(--mono);color:var(--tx);overflow-wrap:anywhere;}
.callout{display:flex;gap:14px;align-items:flex-start;padding:16px 20px;border-radius:14px;border:1px solid var(--line2);background:var(--s1);margin-bottom:6px;}
.callout i{flex:none;width:8px;height:8px;border-radius:50%;background:var(--ac);margin-top:7px;}
.callout.caution i{background:var(--warn);}
.callout.caution{border-color:rgba(230,179,114,.35);}
.callout div{font-size:14px;line-height:1.65;color:var(--tx2);}
.callout b{color:var(--tx);font-weight:600;}
@media (max-width:860px){.verdict{grid-template-columns:1fr;gap:22px;padding:26px}.v-num{text-align:left}.facts{grid-template-columns:1fr 1fr}}

/* Score rows */
.rows{display:grid;gap:20px;margin:10px 0 6px;}
.r-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:9px;}
.r-name{font:500 15px var(--sans);color:var(--tx2);}
.r-pct{font:400 13px var(--mono);color:var(--mu);}
.r-track{height:4px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden;}
.r-fill{height:100%;border-radius:99px;background:#445064;}
.row.lead .r-name{color:var(--tx);font-weight:600;}
.row.lead .r-pct{color:var(--ac);}
.row.lead .r-fill{background:var(--ac);}

/* Inline sections */
.sub{font:400 clamp(26px,2.8vw,34px)/1.1 var(--serif);letter-spacing:-.015em;color:var(--tx);margin:46px 0 6px;}
.sub-s{font-size:14.5px;color:var(--mu);margin-bottom:22px;max-width:620px;line-height:1.6;}
.cap{display:flex;justify-content:space-between;align-items:baseline;margin:0 2px 10px;}
.cap b{font:600 14px var(--sans);color:var(--tx);display:flex;align-items:center;}
.cap span{font:400 12px var(--mono);color:var(--mu);}
.cap.hot b{color:var(--ac);}
.legend{display:flex;align-items:center;gap:14px;margin:22px 0 4px;max-width:520px;}
.legend span{font-size:12.5px;color:var(--mu);white-space:nowrap;}
.legend div{flex:1;height:6px;border-radius:99px;border:1px solid var(--line2);}

/* Grad-CAM */
.cap .n{display:inline-grid;place-items:center;width:19px;height:19px;border-radius:6px;
  background:var(--s3);border:1px solid var(--line2);font:500 10px var(--mono);color:var(--mu);margin-right:9px;}
.cap.hot .n{background:var(--ac-soft);border-color:rgba(138,211,195,.42);color:var(--ac);}
.cam-note{font:400 12px var(--mono);line-height:1.55;color:var(--mu);
  border-left:2px solid var(--line2);padding:2px 0 2px 12px;margin-top:34px;}

/* Performance */
.perf{border:1px solid var(--line);border-radius:18px;overflow:hidden;background:var(--s1);margin-bottom:16px;}
.pf{display:grid;grid-template-columns:1.1fr 1.5fr 1.5fr;gap:30px;align-items:center;padding:22px 26px;border-top:1px solid var(--line);}
.pf:first-child{border-top:none;}
.pf.best{background:linear-gradient(90deg,var(--ac-soft),transparent 70%);}
.pf-name{font:600 16px var(--sans);color:var(--tx);}
.pf-best{display:block;font:400 12px var(--mono);color:var(--ac);margin-top:4px;}
.pf-m .top{display:flex;justify-content:space-between;font-size:12px;color:var(--mu);margin-bottom:8px;}
.pf-m .top b{font:500 13px var(--mono);color:var(--tx);}
@media (max-width:860px){.pf{grid-template-columns:1fr;gap:14px}}

.footer{text-align:center;color:var(--mu);font-size:12.5px;line-height:1.8;padding-top:34px;border-top:1px solid var(--line);margin-top:56px;}
</style>
"""


def ui(markup: str, target=st) -> None:
    """Render raw HTML. Whitespace is collapsed so Markdown never treats it as a code block."""
    target.markdown(re.sub(r"\s*\n\s*", " ", markup.strip()), unsafe_allow_html=True)


def stepper(states):
    steps = [("Upload", "Add one brain MRI image."), ("Classify", "Run the selected model."), ("Explain", "Review scores and Grad-CAM.")]
    out = []
    for i, ((title, sub), state) in enumerate(zip(steps, states), start=1):
        mark = "✓" if state == "done" else str(i)
        out.append(f'<div class="rs {state}"><div class="dot">{mark}</div><div class="rs-t">{title}</div><div class="rs-s">{sub}</div></div>')
    return f'<div class="rail">{"".join(out)}</div>'


def score_rows(items):
    out = []
    for i, (name, pct) in enumerate(items):
        cls = "row lead" if i == 0 else "row"
        out.append(
            f'<div class="{cls}"><div class="r-top"><span class="r-name">{html.escape(str(name))}</span>'
            f'<span class="r-pct">{pct:.1f}%</span></div>'
            f'<div class="r-track"><div class="r-fill" style="width:{max(pct, 0.5):.2f}%"></div></div></div>'
        )
    return f'<div class="rows">{"".join(out)}</div>'


def perf_metric(label, shown, width, best):
    color = "var(--ac)" if best else "#445064"
    return (f'<div class="pf-m"><div class="top"><span>{label}</span><b>{shown}</b></div>'
            f'<div class="r-track"><div class="r-fill" style="width:{max(width, 0.5):.2f}%;background:{color}"></div></div></div>')


def region_name(v_frac: float, h_frac: float) -> str:
    """Turn a normalised peak coordinate into a readable anatomical-ish label."""
    v = "upper" if v_frac < 0.38 else ("lower" if v_frac > 0.62 else "middle")
    h = "left" if h_frac < 0.38 else ("right" if h_frac > 0.62 else "centre")
    if v == "middle" and h == "centre":
        return "centre"
    if v == "middle":
        return h
    if h == "centre":
        return v
    return f"{v} {h}"


@st.cache_resource(show_spinner=False)
def load_checkpoint(path_str: str, device_str: str, mtime: float):
    """Load once per checkpoint file; reruns (slider, tabs) reuse the model."""
    device = torch.device(device_str)
    ckpt = torch.load(path_str, map_location=device)
    model = build_model(ckpt["model_name"], len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, list(ckpt["classes"]), ckpt.get("image_size", 224), ckpt["model_name"]


@st.cache_data(show_spinner=False, max_entries=8)
def compute_gradcam(image_bytes: bytes, path_str: str, mtime: float, device_str: str, class_idx: int, size: int):
    """One Grad-CAM per (image, class, checkpoint). Overlay controls then redraw instantly."""
    device = torch.device(device_str)
    model, _classes, _size, model_name = load_checkpoint(path_str, device_str, mtime)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(image).unsqueeze(0).to(device)

    helper = GradCAM(model, get_target_layer(model, model_name))
    try:
        cam, _ = helper(tensor, class_idx)
    finally:
        helper.close()

    cam = np.asarray(cam, dtype=float)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR), dtype=float) / 255.0
    return cam


CHECKPOINTS = {
    "EfficientNetB0": Path("models/efficientnet_b0.pth"),
    "ResNet50": Path("models/resnet50.pth"),
    "Custom CNN": Path("models/custom_cnn.pth"),
}
MODEL_ARGS = {"EfficientNetB0": "efficientnet_b0", "ResNet50": "resnet50", "Custom CNN": "custom_cnn"}

ui(CSS)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    ui('<div class="sb-h">Model</div>')
    selected = st.selectbox("Model architecture", list(CHECKPOINTS.keys()), label_visibility="collapsed")
    checkpoint_path = CHECKPOINTS[selected]
    has_ckpt = checkpoint_path.exists()
    if has_ckpt:
        ui('<span class="tag live">Checkpoint ready</span>')
    else:
        ui('<span class="tag warn">Checkpoint missing</span>')
        st.caption(f"Expected at {checkpoint_path.as_posix()}")
    ui('<div class="sb-h">Accepted formats</div><span class="fmt">PNG</span><span class="fmt">JPG</span><span class="fmt">JPEG</span><span class="fmt">BMP</span>')
    ui('<div class="sb-h">Intended use</div><div class="note">Research prototype. Not clinically validated. '
       'Do not use for medical decisions.</div>')

# ── Top bar ──────────────────────────────────────────────────────────────────
status = f'<span class="tag live">{html.escape(selected)}</span>' if has_ckpt else f'<span class="tag warn">{html.escape(selected)} not trained</span>'
ui(f"""
<div class="top">
  <div class="logo">
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <circle cx="14" cy="14" r="12.5" stroke="#8ad3c3" stroke-opacity=".55"/>
      <path d="M7.5 15a6.5 6.5 0 0 1 13 0" stroke="#8ad3c3" stroke-width="1.5" stroke-linecap="round"/>
      <circle cx="14" cy="15" r="1.8" fill="#8ad3c3"/>
    </svg>NeuroVision
  </div>
  <div class="top-right">{status}<span class="tag">Research use only</span></div>
</div>
<div class="h1">Brain MRI analysis you can interrogate.</div>
<div class="lede">Upload a scan to see the model's prediction, how it spreads its confidence across classes,
and which regions of the image drove the result.</div>
""")
stepper_slot = st.empty()

# ── Input ────────────────────────────────────────────────────────────────────
left, right = st.columns([1.2, .8], gap="large")
with left:
    uploaded = st.file_uploader("Choose MRI image", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")
    image_pil = None
    if uploaded is not None:
        image_pil = Image.open(uploaded).convert("RGB")
        st.image(image_pil, caption=uploaded.name, use_container_width=True)
with right:
    ui("""
    <div class="panel"><div class="p-title">How to read the output</div>
      <div class="p-copy">Scores come from a softmax layer. They show how the model ranks classes, not calibrated
      clinical probabilities. Grad-CAM highlights image regions that influenced the prediction. It is an approximation
      and does not prove a medically meaningful explanation.</div></div>
    <div class="panel"><div class="p-title">Processing stays local</div>
      <div class="p-copy">The image is analysed by the checkpoint on this machine. Nothing is uploaded elsewhere.</div></div>
    """)

if not has_ckpt:
    st.info(f"No checkpoint found at `{checkpoint_path.as_posix()}`. The interface works, but predictions need a trained model.")
    st.code(f"python train.py --data-dir data --model {MODEL_ARGS[selected]} --epochs 3", language="powershell")

analyzed = uploaded is not None and image_pil is not None and has_ckpt
if analyzed:
    ui(stepper(["done", "done", "done"]), stepper_slot)
elif uploaded is not None:
    ui(stepper(["done", "active", "idle"]), stepper_slot)
else:
    ui(stepper(["active", "idle", "idle"]), stepper_slot)

# ── Results ──────────────────────────────────────────────────────────────────
if analyzed:
    st.divider()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.spinner("Analyzing image..."):
        model, classes, size, model_name = load_checkpoint(str(checkpoint_path), str(device), checkpoint_path.stat().st_mtime)
        transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(scores))

    order = np.argsort(scores)[::-1]
    top_pct = float(scores[idx] * 100)
    margin = float((scores[order[0]] - scores[order[1]]) * 100) if len(scores) > 1 else 100.0
    runner = classes[order[1]] if len(scores) > 1 else None

    ui('<div class="sec">Analysis results</div>'
       '<div class="sec-s">A dataset classification from your trained model. It is not a medical diagnosis.</div>')

    ui(f"""
    <div class="verdict">
      <div><div class="v-k">Predicted dataset class</div><div class="v-class">{html.escape(str(classes[idx]))}</div></div>
      <div class="v-num"><div class="v-k">Highest model score</div><div class="v-pct">{top_pct:.1f}<small>%</small></div></div>
    </div>
    <div class="meter" style="margin:-4px 4px 26px"><div class="m-track"><div class="m-fill" style="width:{top_pct:.2f}%"></div></div>
      <div class="m-tick"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div></div>
    <div class="facts">
      <div class="fact"><span>Model</span><b>{html.escape(str(model_name))}</b></div>
      <div class="fact"><span>Input size</span><b>{size} × {size}</b></div>
      <div class="fact"><span>Lead over next class</span><b>{margin:.1f} pts</b></div>
      <div class="fact"><span>Device</span><b>{str(device).upper()}</b></div>
    </div>
    """)

    if top_pct < 60 or margin < 20:
        nxt = f" Next closest: <b>{html.escape(str(runner))}</b>." if runner is not None else ""
        ui(f'<div class="callout caution"><i></i><div><b>Low separation.</b> The top classes are close '
           f'({margin:.1f} points apart).{nxt} Treat this prediction as uncertain.</div></div>')
    else:
        ui(f'<div class="callout"><i></i><div><b>Clear separation.</b> The top class leads by {margin:.1f} points. '
           f'This describes the model output only. Softmax scores can be overconfident.</div></div>')

    score_df = pd.DataFrame({"Dataset class": classes, "Model score (%)": (scores * 100).round(2)}).sort_values("Model score (%)", ascending=False)

    # Class scores
    ui('<div class="sub">Class scores</div><div class="sub-s">How the model distributed its confidence, highest first.</div>')
    c1, c2 = st.columns([1.1, .9], gap="large")
    with c1:
        ui(f'<div class="panel" style="margin-bottom:0">{score_rows([(classes[i], float(scores[i] * 100)) for i in order])}</div>')
    with c2:
        st.dataframe(score_df, hide_index=True, use_container_width=True)

    # ── Grad-CAM — inline, part of the same page flow ────────────────────────
    ui('<div class="sub">Where the model looked</div>'
       f'<div class="sub-s">Grad-CAM for the predicted class, <b style="color:var(--tx)">{html.escape(str(classes[idx]))}</b>. '
       'Brighter regions had more influence on the score. It is an approximation, not a clinical explanation.</div>')

    if model_name in ("resnet50", "efficientnet_b0"):
        ctl1, ctl2, ctl3 = st.columns([1, 1, 1.6], gap="medium")
        with ctl1:
            alpha = st.slider("Overlay strength", 0.1, 0.9, 0.5, 0.05)
        with ctl2:
            cmap_name = st.selectbox("Colormap", ["magma", "inferno", "turbo", "jet"])
        with ctl3:
            ui('<div class="cam-note">Heat map is computed once per image and class, so these controls redraw instantly.</div>')

        cam = compute_gradcam(
            uploaded.getvalue(), str(checkpoint_path), checkpoint_path.stat().st_mtime,
            str(device), idx, int(size),
        )
        base_arr = np.asarray(image_pil.resize((size, size)), dtype=float) / 255.0
        cmap = plt.get_cmap(cmap_name)
        heat = cmap(cam)[..., :3]
        overlay = np.clip(base_arr * (1 - alpha) + heat * alpha, 0, 1)
        to_u8 = lambda a: (a * 255).astype(np.uint8)

        g1, g2, g3 = st.columns(3, gap="medium")
        with g1:
            ui('<div class="cap"><b><span class="n">1</span>Original</b><span>input</span></div>')
            st.image(to_u8(base_arr), use_container_width=True)
        with g2:
            ui('<div class="cap hot"><b><span class="n">2</span>Overlay</b><span>model attention</span></div>')
            st.image(to_u8(overlay), use_container_width=True)
        with g3:
            ui('<div class="cap"><b><span class="n">3</span>Attention map</b><span>heatmap only</span></div>')
            st.image(to_u8(heat), use_container_width=True)

        stops = ", ".join(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b, _ in (cmap(t) for t in np.linspace(0, 1, 8)))
        ui(f'<div class="legend"><span>Lower influence</span><div style="background:linear-gradient(90deg,{stops})"></div><span>Higher influence</span></div>')

        peak_y, peak_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
        peak = region_name(peak_y / max(cam.shape[0] - 1, 1), peak_x / max(cam.shape[1] - 1, 1))
        hot_area = float((cam > 0.5).mean() * 100)
        ui(f"""
        <div class="facts" style="margin-top:22px">
          <div class="fact"><span>Peak influence</span><b>{html.escape(peak)}</b></div>
          <div class="fact"><span>Highlighted area</span><b>{hot_area:.1f}%</b></div>
          <div class="fact"><span>Overlay</span><b>{html.escape(cmap_name)} · {alpha:.2f}</b></div>
          <div class="fact"><span>Method</span><b>Grad-CAM</b></div>
        </div>
        """)
        ui('<div class="cam-note">Grad-CAM weights the final convolutional feature maps by the gradient of the '
           'predicted class. Bright areas are where the score was most sensitive — they are not proof of disease.</div>')
    else:
        st.info("Grad-CAM is available for ResNet50 and EfficientNetB0.")

    # Export
    ui('<div class="sub">Export</div><div class="sub-s">Save this analysis for your notes or lab records.</div>')
    report = f"NeuroVision — Research Prototype\nImage: {uploaded.name}\nModel: {model_name}\nPredicted dataset class: {classes[idx]}\nModel scores:\n"
    report += "\n".join(f"- {c}: {s:.4f}" for c, s in zip(classes, scores))
    report += "\n\nNot a medical diagnostic tool."
    d1, d2, _ = st.columns([1, 1, 1.4])
    d1.download_button("Download report (.txt)", report, file_name="neurovision_report.txt", mime="text/plain", use_container_width=True)
    d2.download_button("Download scores (.csv)", score_df.to_csv(index=False), file_name="neurovision_scores.csv", mime="text/csv", use_container_width=True)
    with st.expander("Preview report"):
        st.code(report, language="text")

# ── Model performance ────────────────────────────────────────────────────────
st.divider()
ui('<div class="sec">Model performance</div>'
   '<div class="sec-s">Saved evaluation metrics for the models you have trained and evaluated.</div>')
metric_files = {
    "EfficientNetB0": "outputs/efficientnet_b0_metrics.json",
    "ResNet50": "outputs/resnet50_metrics.json",
    "Custom CNN": "outputs/custom_cnn_metrics.json",
}
comparison = []
for model_label, file_path in metric_files.items():
    path = Path(file_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            comparison.append({"Model": model_label, "Accuracy (%)": round(metrics["accuracy"] * 100, 2), "Macro F1": round(metrics["macro_f1"], 4)})
        except (OSError, ValueError, KeyError, TypeError):
            pass

if comparison:
    df = pd.DataFrame(comparison)
    best = df.sort_values("Macro F1", ascending=False).iloc[0]
    rows = []
    for r in comparison:
        is_best = r["Model"] == best["Model"]
        note = '<span class="pf-best">Best macro F1</span>' if is_best else ""
        acc, f1 = r["Accuracy (%)"], r["Macro F1"]
        name = html.escape(r["Model"])
        row_cls = "pf best" if is_best else "pf"
        acc_html = perf_metric("Accuracy", f"{acc:.2f}%", acc, is_best)
        f1_html = perf_metric("Macro F1", f"{f1:.4f}", f1 * 100, is_best)
        rows.append(f'<div class="{row_cls}"><div><span class="pf-name">{name}</span>{note}</div>{acc_html}{f1_html}</div>')
    ui(f'<div class="perf">{"".join(rows)}</div>')
    st.dataframe(df, hide_index=True, use_container_width=True)
else:
    ui('<div class="panel"><div class="p-title">No evaluation metrics yet</div>'
       '<div class="p-copy">Run evaluation on your trained models. Accuracy and macro F1 will appear here.</div></div>')

ui('<div class="footer">NeuroVision, explainable brain MRI classification.<br>'
   'Educational and research prototype. Not clinically validated. Do not use for medical decisions.</div>')
