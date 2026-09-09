"""
Streamlit app: upload a brain MRI (four co-registered modalities) and get an
automatic glioblastoma segmentation — tumor location overlaid on the scan, or a
"no tumor detected" message.

Run from the repo root:
    streamlit run app/streamlit_app.py

The trained checkpoint path defaults to the bundled fp16 weights
(app/weights/), or BRATS_CHECKPOINT (env); override it in the sidebar.

NOTE: this file is presentation only. Every model/data call below goes
through `app.inference` (imported as `inf`) unchanged — no inference,
preprocessing, checkpoint, or dataset logic is redefined here.
"""

import base64
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import inference as inf  # noqa: E402

DEFAULT_CKPT = os.environ.get("BRATS_CHECKPOINT", inf.DEFAULT_CHECKPOINT)
DEFAULT_DATA_ROOT = os.environ.get(
    "DATA_ROOT", r"U:\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
)
GITHUB_URL = "https://github.com/Utkarsh4518/Glioblastoma-Brain-Tumor-Detection"
REPORT_URL = f"{GITHUB_URL}/blob/main/report/main.pdf"
FIGURES_DIR = REPO_ROOT / "report" / "figures"

EXAMPLE_META = {
    "BraTS20_Training_341": ("Small tumor", "small"),
    "BraTS20_Training_049": ("Medium tumor", "medium"),
    "BraTS20_Training_001": ("Large tumor", "large"),
}
MODALITY_SUBLABELS = {
    "t1": "Native T1-weighted",
    "t1ce": "Contrast-enhanced",
    "t2": "T2-weighted",
    "flair": "Fluid-attenuated IR",
}

st.set_page_config(page_title="Glioblastoma Tumor Detector", page_icon="🧠", layout="wide")

# ============================================================================
# Style: restrained "research lab" palette — deep navy/near-black text on
# white / very light cool-gray, thin borders, soft shadows, rounded cards,
# small blue/cyan/pink accents used sparingly. Only Streamlit's own widget
# classes are targeted for the interactive controls (functionality is
# untouched) — everything else is plain additive HTML for layout only.
# ============================================================================
st.markdown(
    """
    <style>
    :root {
        --ink: #e6edf3;
        --ink-soft: #9aa7b8;
        --bg-soft: #171c26;
        --card-bg: #161b22;
        --border: #262d3a;
        --accent: #4c8dfa;
        --accent-soft: #14223b;
        --cyan: #22b8cf;
        --pink: #ec4899;
        --shadow: 0 1px 2px rgba(0,0,0,.4), 0 6px 18px rgba(0,0,0,.45);
        --radius: 14px;
    }
    /* extra top padding clears Streamlit's fixed header bar, which otherwise
       hides small elements (like the eyebrow line) sitting at the very top */
    .block-container { max-width: 1180px; padding-top: 3.2rem; padding-bottom: 3rem; }

    /* ---- generic section rhythm ---- */
    .section-gap { margin-top: 2.6rem; }
    .eyebrow {
        font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
        color: var(--cyan); margin-bottom: .5rem;
    }
    .section-title { font-size: 1.5rem; font-weight: 700; color: var(--ink); margin: 0 0 .3rem 0; }
    .section-sub { color: var(--ink-soft); font-size: .95rem; margin-bottom: 1.2rem; }

    /* ---- hero ---- */
    .hero-title { font-size: 2.5rem; line-height: 1.1; font-weight: 800; color: var(--ink); margin: 0 0 .6rem 0; }
    .hero-tagline { font-size: 1.05rem; color: var(--accent); font-weight: 600; margin-bottom: .9rem; }
    .hero-body { color: var(--ink-soft); font-size: .98rem; line-height: 1.55; max-width: 46ch; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 8px; margin-top: 1.1rem;
        padding: 6px 14px; border-radius: 999px; background: var(--accent-soft);
        border: 1px solid #24406e; color: var(--accent); font-weight: 600; font-size: .82rem;
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #16a34a; display: inline-block; }
    .status-dot.warn { background: #d97706; }

    .hero-visual {
        border-radius: var(--radius); border: 1px solid var(--border); background: #0e1526;
        box-shadow: var(--shadow); padding: 22px; height: 100%;
        display: flex; flex-direction: column; justify-content: space-between;
    }
    .hero-visual .cap { color: #93a3c2; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 14px; }
    .slice-stack { position: relative; height: 150px; margin: 6px 0 18px 28px; }
    .slice-stack .slab {
        position: absolute; width: 150px; height: 110px; border-radius: 10px;
        border: 1px solid rgba(255,255,255,.14); background: linear-gradient(160deg,#1a2440,#101728);
    }
    .slice-stack .slab:nth-child(1) { transform: translate(0px, 30px); opacity: .55; }
    .slice-stack .slab:nth-child(2) { transform: translate(14px, 18px); opacity: .75; }
    .slice-stack .slab:nth-child(3) { transform: translate(28px, 6px); opacity: .92; }
    .slice-stack .slab:nth-child(4) { transform: translate(42px, -6px); }
    .slice-stack .tumor-dot {
        position: absolute; width: 22px; height: 22px; border-radius: 50%; top: 34px; left: 78px;
        background: radial-gradient(circle at 35% 35%, #f472b6, #db2777 70%);
        box-shadow: 0 0 0 6px rgba(219,39,119,.15);
    }
    .hero-visual .flow { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; color: #cfd8ec; font-size: .78rem; font-weight: 600; }
    .hero-visual .flow .arrow { color: #4c5a7a; }
    .hero-visual .modtags { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 12px; }
    .modtag { font-size: .7rem; font-weight: 600; color: #a7b4d1; border: 1px solid #2a3555; border-radius: 999px; padding: 3px 10px; }

    /* ---- info / pipeline cards (pure HTML, no live widgets inside) ---- */
    .card-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    @media (max-width: 900px) { .card-grid { grid-template-columns: 1fr; } }
    .info-card {
        border: 1px solid var(--border); border-radius: var(--radius); background: var(--card-bg);
        box-shadow: var(--shadow); padding: 18px 20px; height: 100%;
    }
    .info-card .tag { font-size: .68rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--cyan); }
    .info-card h4 { margin: .3rem 0 .3rem 0; font-size: 1.08rem; color: var(--ink); }
    .info-card p { margin: 0; color: var(--ink-soft); font-size: .88rem; line-height: 1.5; }

    .pipeline { display: flex; align-items: stretch; gap: 8px; flex-wrap: wrap; }
    .pipe-step {
        flex: 1 1 190px; border: 1px solid var(--border); border-radius: var(--radius);
        background: var(--card-bg); box-shadow: var(--shadow); padding: 16px 16px; position: relative;
    }
    .pipe-step .num { font-size: .72rem; font-weight: 800; color: #fff; background: var(--accent);
        width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 10px; }
    .pipe-step h5 { margin: 0 0 4px 0; font-size: .92rem; letter-spacing: .02em; color: var(--ink); }
    .pipe-step p { margin: 0; font-size: .8rem; color: var(--ink-soft); }
    .pipe-arrow { display: flex; align-items: center; justify-content: center; color: #b6bfd1; font-size: 1.2rem; flex: 0 0 auto; }
    @media (max-width: 900px) { .pipe-arrow { display: none; } }

    /* ---- example pills legend dots ---- */
    .size-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
    .size-dot.small { background:#0891b2; }
    .size-dot.medium { background:#d97706; }
    .size-dot.large { background:#db2777; }

    /* ---- results ---- */
    .result-banner {
        border-radius: var(--radius); padding: 16px 20px; border: 1px solid var(--border);
        display: flex; align-items: center; gap: 12px; margin-bottom: 14px;
    }
    .result-banner.tumor { background: #241a0d; border-color: #6b4415; }
    .result-banner.clear { background: #0f2117; border-color: #1e5631; }
    .result-banner .headline { font-weight: 700; font-size: 1.05rem; color: var(--ink); }
    .result-banner .sub { color: var(--ink-soft); font-size: .85rem; }

    .metric-card {
        border: 1px solid var(--border); border-radius: 12px; background: var(--card-bg); box-shadow: var(--shadow);
        padding: 12px 14px; margin-bottom: 10px;
    }
    .metric-card .lbl { font-size: .82rem; color: var(--ink-soft); display: flex; align-items: center; gap: 7px; }
    .metric-card .val { font-size: 1.4rem; font-weight: 700; color: var(--ink); margin-top: 2px; }
    .swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }

    .img-card { border: 1px solid var(--border); border-radius: var(--radius); background: #0b1120;
        box-shadow: var(--shadow); padding: 14px 14px 12px 14px; }
    .img-card.hero-viz { padding: 18px 18px 16px 18px; }
    .img-card .img-tag {
        font-size: .68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
        color: var(--cyan); margin-bottom: 10px;
    }
    .img-card img { border-radius: 8px; width: 100%; display: block; aspect-ratio: 1 / 1; object-fit: cover; }
    .img-card .cap { text-align: center; color: #cbd5e1; font-size: .78rem; font-weight: 600; margin-top: 10px; }
    .placeholder-panel {
        border: 1px dashed var(--border); border-radius: var(--radius); background: var(--bg-soft);
        padding: 40px 20px; text-align: center; color: var(--ink-soft); font-size: .9rem;
    }

    /* ---- input -> output flow connector (shown right before results) ---- */
    .flow-strip {
        display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
        padding: 14px 20px; border: 1px solid var(--border); border-radius: 12px;
        background: var(--card-bg); margin: 1.2rem 0 1.6rem 0;
    }
    .flow-strip .fnode { display: flex; flex-direction: column; line-height: 1.25; }
    .flow-strip .fnode .ftitle { font-size: .82rem; font-weight: 700; color: var(--ink); letter-spacing: .02em; }
    .flow-strip .fnode .fsub { font-size: .72rem; color: var(--ink-soft); }
    .flow-strip .farrow { color: var(--ink-soft); font-size: 1.1rem; flex: 0 0 auto; }

    .about-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    @media (max-width: 900px) { .about-grid { grid-template-columns: repeat(2, 1fr); } }
    .about-item { border: 1px solid var(--border); border-radius: 12px; background: var(--card-bg); padding: 14px 16px; }
    .about-item .k { font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; color: var(--cyan); font-weight: 700; }
    .about-item .v { font-size: .95rem; color: var(--ink); font-weight: 600; margin-top: 3px; }

    .snapshot-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; align-items: stretch; }
    @media (max-width: 900px) { .snapshot-grid { grid-template-columns: 1fr; } }
    .snapshot-card {
        border: 1px solid var(--border); border-radius: var(--radius); background: var(--card-bg);
        box-shadow: var(--shadow); overflow: hidden; height: 100%; display: flex; flex-direction: column;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .snapshot-card:hover { transform: translateY(-3px); box-shadow: 0 2px 4px rgba(0,0,0,.4), 0 12px 26px rgba(0,0,0,.5); }
    .snapshot-img-wrap { height: 190px; background: #0b1120; display: flex; align-items: center; justify-content: center; overflow: hidden; }
    .snapshot-img-wrap img { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
    .snapshot-card .cap { padding: 10px 14px 14px 14px; font-size: .85rem; font-weight: 600; color: var(--ink); margin-top: auto; }

    .app-footer { text-align: center; color: var(--ink-soft); font-size: .82rem; padding-top: 8px; }
    .app-footer .line1 { font-weight: 600; color: var(--ink); }

    hr.thin { border: none; border-top: 1px solid var(--border); margin: 2.2rem 0; }

    /* ---- polish a few native Streamlit widgets to match the palette ---- */
    div.stButton > button[kind="primary"] { height: 2.9em; font-size: 1.02em; font-weight: 700; border-radius: 10px; }
    div[data-testid="stExpander"] { border-radius: 12px; border: 1px solid var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Small display-only helpers (no model/data logic — pure presentation)
# ============================================================================
def _swatch(rgb: tuple[int, int, int]) -> str:
    return f"<span class='swatch' style='background:rgb{rgb}'></span>"


def _legend_html() -> str:
    chips = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:6px;margin-right:20px;font-size:.85rem;color:var(--ink-soft)'>"
        f"{_swatch(inf.CLASS_COLORS[c])} {inf.CLASS_NAMES[c]}</span>"
        for c in (1, 2, 3)
    )
    return f"<div style='margin-top:.6rem'>{chips}</div>"


def _b64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _np_to_b64_png(arr: np.ndarray) -> str:
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_card_html(src_b64: str, caption: str, tag: str | None = None, hero: bool = False) -> str:
    tag_html = f"<div class='img-tag'>{tag}</div>" if tag else ""
    cls = "img-card hero-viz" if hero else "img-card"
    return (
        f"<div class='{cls}'>{tag_html}<img src='data:image/png;base64,{src_b64}'/>"
        f"<div class='cap'>{caption}</div></div>"
    )


@st.cache_resource(show_spinner=False)
def _get_model(ckpt: str, device_pref: str):
    device = inf.get_device(device_pref)
    model, epoch = inf.load_model(ckpt, device)
    return model, epoch, device


@st.cache_data(show_spinner=False)
def _list_example_subjects(data_root: str) -> dict:
    """Map subject_id -> {modality: path, 'seg': path} using the project loader's discovery."""
    from data.nifti_brats_dataset import NiftiBraTSDataset
    ds = NiftiBraTSDataset(root=data_root, patch_size=None)
    return {pid: ds._paths[pid] for pid in ds.subjects}


def _save_upload(uploaded) -> str:
    suffix = ".nii.gz" if uploaded.name.endswith(".gz") else ".nii"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


def _modality_preview_grid(volumes: list[np.ndarray]) -> None:
    """2x2 real modality preview from the loaded (pre-inference) volumes."""
    cols = st.columns(4)
    for col, mod, vol in zip(cols, inf.MODALITIES, volumes):
        mid = vol.shape[0] // 2
        panel = inf.make_overlay(vol[mid], np.zeros_like(vol[mid], dtype=np.int64))
        with col:
            st.markdown(
                _img_card_html(
                    _np_to_b64_png(panel),
                    MODALITY_SUBLABELS.get(mod, ""),
                    tag=inf.MODALITY_LABELS[mod],
                ),
                unsafe_allow_html=True,
            )


# ============================================================================
# 1. HERO
# ============================================================================
hero_left, hero_right = st.columns([3, 2], gap="large")
with hero_left:
    st.markdown(
        """
        <div class="eyebrow">Medical Imaging &nbsp;•&nbsp; Deep Learning &nbsp;•&nbsp; Research Demonstrator</div>
        <div class="hero-title">Glioblastoma Tumor Detection</div>
        <div class="hero-tagline">AI-assisted segmentation from multimodal brain MRI</div>
        <div class="hero-body">
            This research demonstrator uses a 3D U-Net trained on the BraTS 2020 dataset to
            identify and segment glioblastoma sub-regions from multimodal MRI scans —
            necrotic core, edema, and enhancing tumor.
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_slot = st.empty()
    status_slot.markdown(
        "<div class='status-pill'><span class='status-dot warn'></span>Loading model…</div>",
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown(
        """
        <div class="hero-visual">
            <div>
                <div class="cap">Volumetric MRI Input</div>
                <div class="slice-stack">
                    <div class="slab"></div><div class="slab"></div><div class="slab"></div><div class="slab"></div>
                    <div class="tumor-dot"></div>
                </div>
                <div class="modtags">
                    <span class="modtag">T1</span><span class="modtag">T1Gd</span>
                    <span class="modtag">T2</span><span class="modtag">FLAIR</span>
                </div>
            </div>
            <div class="flow">
                <span>MRI</span><span class="arrow">→</span><span>3D U-Net</span>
                <span class="arrow">→</span><span>Segmentation</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ============================================================================
# 2. WHAT IS THIS PROJECT?
# ============================================================================
st.markdown(
    """
    <div class="section-title">From MRI scans to tumor segmentation</div>
    <div class="section-sub">A short overview of the data, the model, and what the output represents.</div>
    <div class="card-grid">
        <div class="info-card">
            <div class="tag">Input</div>
            <h4>Multimodal MRI</h4>
            <p><b>T1 • T1Gd • T2 • FLAIR</b> — four MRI sequences that each highlight different
            tissue properties. Combining them gives the model complementary information about
            tumor tissue that no single scan provides alone.</p>
        </div>
        <div class="info-card">
            <div class="tag">Model</div>
            <h4>3D U-Net</h4>
            <p><b>Volumetric deep learning</b> — an encoder–decoder network with skip connections
            that processes the full 3D scan at once, predicting a tumor class for every voxel
            rather than working slice by slice.</p>
        </div>
        <div class="info-card">
            <div class="tag">Output</div>
            <h4>Tumor Sub-regions</h4>
            <p><b>Core • Edema • Enhancing tumor</b> — the segmentation separates the necrotic/
            non-enhancing core, the surrounding edema, and the actively enhancing tumor tissue.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

# ============================================================================
# 3. HOW IT WORKS
# ============================================================================
st.markdown(
    """
    <div class="section-title">How it works</div>
    <div class="section-sub">The scan passes through four stages to produce a 3D tumor mask.</div>
    <div class="pipeline">
        <div class="pipe-step"><div class="num">1</div><h5>MRI INPUT</h5><p>Four registered MRI modalities</p></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="num">2</div><h5>PREPROCESSING</h5><p>Normalize and prepare the volumetric scan</p></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="num">3</div><h5>3D U-NET</h5><p>Deep-learning segmentation</p></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="num">4</div><h5>3D MASK</h5><p>Tumor sub-region prediction</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown("### 🧠 Glioblastoma Tumor Detector")
    st.markdown("**About**")
    st.markdown(
        "A 3D U-Net segments glioblastoma sub-regions — necrotic core, edema, "
        "and enhancing tumor — from four co-registered MRI modalities "
        "(T1, T1Gd, T2, FLAIR)."
    )
    st.markdown(f"📄 [Full Research Report]({REPORT_URL})  \n💻 [Source Code / GitHub]({GITHUB_URL})")
    st.divider()
    sidebar_status_slot = st.empty()
    st.divider()
    with st.expander("🛠️ Developer / Advanced"):
        ckpt = st.text_input("Checkpoint path", value=DEFAULT_CKPT)
        device_pref = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        data_root = st.text_input("Local BraTS data root (optional)", value=DEFAULT_DATA_ROOT)

# ----- model load: unchanged logic/order from the original app -----
if not Path(ckpt).exists():
    status_slot.markdown(
        "<div class='status-pill' style='background:#2a1215;border-color:#5c2630;color:#f87171'>"
        "<span class='status-dot' style='background:#ef4444'></span>Model checkpoint not found</div>",
        unsafe_allow_html=True,
    )
    st.error(
        f"Model checkpoint not found at:\n\n`{ckpt}`\n\n"
        "Set the correct path in the sidebar under **Developer / Advanced** (or the "
        "`BRATS_CHECKPOINT` environment variable). This is the trained `*_best.pt` file."
    )
    st.stop()

model, epoch, device = _get_model(ckpt, device_pref)

status_slot.markdown(
    f"<div class='status-pill'><span class='status-dot'></span>Research Model Ready — "
    f"epoch {epoch} · {device.type.upper()}</div>",
    unsafe_allow_html=True,
)
sidebar_status_slot.markdown(
    f"""
    <div class="metric-card" style="margin-bottom:0">
        <div class="lbl"><span class="status-dot"></span> MODEL READY</div>
        <div class="val" style="font-size:1rem">Epoch {epoch}</div>
        <div class="lbl">Device: {device.type.upper()}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# 4. ANALYZE A SCAN
# ============================================================================
st.markdown(
    """
    <div class="section-title">Analyze a Scan</div>
    <div class="section-sub">Explore an example case or provide your own BraTS-compatible MRI data.</div>
    """,
    unsafe_allow_html=True,
)

volumes = None          # list of 4 arrays [t1, t1ce, t2, flair]
gt = None                # optional ground-truth label volume
source_label = ""

with st.container(border=True):
    mode = st.segmented_control(
        "Input source",
        ["✨ Try an example", "📤 Upload your scan"],
        default="✨ Try an example",
        label_visibility="collapsed",
    )

    if mode == "📤 Upload your scan":
        st.markdown(
            "Upload the **four co-registered MRI modalities** (BraTS-style, "
            "skull-stripped, 1 mm isotropic). A single scan cannot be segmented "
            "reliably — the model needs all four."
        )
        cols = st.columns(4)
        uploads = {}
        for col, mod in zip(cols, inf.MODALITIES):
            with col:
                st.markdown(f"**{inf.MODALITY_LABELS[mod]}**")
                uploads[mod] = st.file_uploader(
                    inf.MODALITY_LABELS[mod], type=["nii", "gz"], key=mod, label_visibility="collapsed"
                )
        if all(uploads[m] is not None for m in inf.MODALITIES):
            try:
                volumes = [inf.load_nifti(_save_upload(uploads[m])) for m in inf.MODALITIES]
                source_label = "uploaded scan"
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the uploaded files: {e}")

    else:  # Try an example — bundled with the app, so this always works (incl. on the cloud)
        bundled = inf.list_bundled_examples()
        if not bundled:
            st.error("No bundled examples found under app/examples/.")
        else:
            st.caption(
                "Example cases are provided to demonstrate how segmentation changes "
                "across tumor burden."
            )
            pid = st.pills(
                "Example subject",
                list(bundled.keys()),
                format_func=lambda p: EXAMPLE_META.get(p, (p, ""))[0],
                default=list(bundled.keys())[0],
                required=True,
            )
            show_gt = st.toggle("Show ground truth for comparison", value=True)
            if pid:
                volumes, gt_full = inf.load_bundled_example(pid)
                label, _ = EXAMPLE_META.get(pid, (pid, ""))
                source_label = f"{label} — {pid}"
                gt = gt_full if show_gt else None

        with st.expander("Advanced: browse a full local BraTS dataset instead"):
            st.caption(
                "Only useful if you have the full BraTS2020 dataset on this machine "
                "(not available on the hosted demo)."
            )
            local_root = st.text_input("Local BraTS data root", value=data_root, key="local_root")
            if Path(local_root).exists():
                local_subjects = _list_example_subjects(local_root)
                local_pid = st.selectbox("Local subject", list(local_subjects.keys()), key="local_pid")
                if st.button("Use this local subject"):
                    paths = local_subjects[local_pid]
                    volumes = [inf.load_nifti(paths[m]) for m in inf.MODALITIES]
                    source_label = local_pid
                    if "seg" in paths:
                        seg = inf.load_nifti(paths["seg"]).astype(np.int64)
                        seg[seg == 4] = 3
                        gt = seg
            else:
                st.caption(f"Path not found: `{local_root}`")

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    analyze = st.button(
        "🔍 Analyze Scan", type="primary", use_container_width=True, disabled=(volumes is None)
    )
    if volumes is None:
        st.caption("Select an example or upload a scan above to enable analysis.")

# ============================================================================
# 5. MRI MODALITY VISUALIZATION
# ============================================================================
st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-title">Multimodal MRI</div>
    <div class="section-sub">The four input modalities for the selected scan.</div>
    """,
    unsafe_allow_html=True,
)
if volumes is not None:
    _modality_preview_grid(volumes)
else:
    st.markdown(
        "<div class='placeholder-panel'>Modality preview available after scan loading.</div>",
        unsafe_allow_html=True,
    )

# ============================================================================
# 6 & 7. RESULTS + GROUND TRUTH COMPARISON
# ============================================================================
if volumes is not None and analyze:
    with st.status("Analysis in progress", expanded=True) as status_box:
        st.write("Preparing volumetric MRI")
        image = inf.preprocess(volumes)             # (4, D, H, W), z-scored + cropped
        st.write("Running 3D U-Net")
        pred = inf.predict(model, image, device)    # (D, H, W)
        st.write("Generating segmentation")
        stats = inf.summarize(pred)
        status_box.update(label="Analysis complete", state="complete", expanded=False)

    # ---- input -> output connector, bridging the MRI preview above to the result below ----
    st.markdown(
        """
        <div class="flow-strip">
            <div class="fnode"><span class="ftitle">MULTIMODAL MRI</span><span class="fsub">T1 • T1Gd • T2 • FLAIR</span></div>
            <div class="farrow">→</div>
            <div class="fnode"><span class="ftitle">3D U-NET</span><span class="fsub">Volumetric segmentation</span></div>
            <div class="farrow">→</div>
            <div class="fnode"><span class="ftitle">SEGMENTATION MASK</span><span class="fsub">Tumor sub-regions</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="section-title">Segmentation Results</div>
        <div class="section-sub">AI-generated tumor sub-region segmentation · Case: {source_label}</div>
        """,
        unsafe_allow_html=True,
    )

    flair = image[inf.MODALITIES.index("flair")]    # (D, H, W), z-scored (fine for display)
    gt_crop = None
    if gt is not None:
        gt_crop = inf._center_crop(gt[np.newaxis].astype(np.float32))[0].astype(np.int64)
        # Default to the ground truth's peak slice, not the prediction's: if the
        # model has a false-positive region larger than the true tumor elsewhere
        # in the volume, defaulting to the prediction's peak would spotlight that
        # false positive instead of a meaningful ground-truth comparison.
        default_slice = inf.best_tumor_slice(gt_crop)
    else:
        default_slice = inf.best_tumor_slice(pred)

    res_left, res_right = st.columns([2.2, 1], gap="large")
    with res_left:
        s = st.slider("Axial slice", 0, pred.shape[0] - 1, default_slice)
        pred_panel = inf.make_overlay(flair[s], pred[s])
        st.markdown(
            _img_card_html(_np_to_b64_png(pred_panel), f"Axial slice {s}", tag="MODEL PREDICTION", hero=True),
            unsafe_allow_html=True,
        )

    with res_right:
        if not stats["tumor_present"]:
            st.markdown(
                "<div class='result-banner clear'><div>"
                "<div class='headline'>✅ No tumor detected</div>"
                "<div class='sub'>No significant tumor tissue was found in this scan.</div>"
                "</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='result-banner tumor'><div>"
                f"<div class='headline'>⚠️ Tumor detected</div>"
                f"<div class='sub'>{stats['total_ml']} mL total predicted volume</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='font-weight:700;color:var(--ink);margin-bottom:.5rem'>Prediction Summary</div>",
                unsafe_allow_html=True,
            )
            for cls in (3, 2, 1):  # ET, ED, NCR
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="lbl">{_swatch(inf.CLASS_COLORS[cls])} {inf.CLASS_NAMES[cls]}</div>
                        <div class="val">{stats['per_class_ml'][cls]} mL</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ---- Segmentation Legend: immediately below the main visualization ----
    st.markdown(
        f"""
        <div class="section-title" style="font-size:1.05rem;margin-top:.4rem">Segmentation Legend</div>
        <div class="section-sub" style="margin-bottom:.3rem">
            The overlay highlights the predicted tumor sub-regions within the MRI volume.
        </div>
        {_legend_html()}
        """,
        unsafe_allow_html=True,
    )

    if gt_crop is not None:
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-title">Prediction vs Ground Truth</div>
            <div class="section-sub">Model prediction against the expert-annotated ground truth, at the same slice.</div>
            """,
            unsafe_allow_html=True,
        )
        gtv1, gtv2 = st.columns(2, gap="large")
        with gtv1:
            st.markdown(
                _img_card_html(_np_to_b64_png(inf.make_overlay(flair[s], pred[s])), f"Axial slice {s}", tag="MODEL PREDICTION"),
                unsafe_allow_html=True,
            )
        with gtv2:
            st.markdown(
                _img_card_html(_np_to_b64_png(inf.make_overlay(flair[s], gt_crop[s])), f"Axial slice {s}", tag="GROUND TRUTH"),
                unsafe_allow_html=True,
            )
        st.markdown(_legend_html(), unsafe_allow_html=True)

    with st.expander("ℹ️ How this works"):
        st.markdown(
            "1. The four MRI modalities are stacked and z-score normalized per-modality.\n"
            "2. A 3D U-Net (≈19M parameters) maps the 4-channel volume to per-voxel class scores.\n"
            "3. Each voxel is assigned its most likely class: background, necrotic/non-enhancing "
            "core, edema, or enhancing tumor.\n"
            "4. Tumor is reported as *detected* when enough foreground voxels are predicted; "
            "the sub-region volumes are voxel counts converted to mL (1 mm³ isotropic).\n\n"
            f"Trained and evaluated on the BraTS 2020 high-grade-glioma (glioblastoma) subjects — "
            f"see the [full report]({REPORT_URL}) for methodology and results."
        )

# ============================================================================
# 8. ABOUT THE MODEL
# ============================================================================
st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
st.markdown("<hr class='thin'/>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-title">About the Model</div>
    <div class="about-grid">
        <div class="about-item"><div class="k">Architecture</div><div class="v">3D U-Net</div></div>
        <div class="about-item"><div class="k">Dataset</div><div class="v">BraTS 2020</div></div>
        <div class="about-item"><div class="k">Input</div><div class="v">T1 • T1Gd • T2 • FLAIR</div></div>
        <div class="about-item"><div class="k">Task</div><div class="v">3D tumor sub-region segmentation</div></div>
    </div>
    <div class="section-sub" style="margin-top:14px">
        This system is a research and educational demonstrator built for a bachelor's
        thesis project. It is <b>not a clinical diagnostic tool</b> and has not been
        validated for medical use.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# 9. RESEARCH SNAPSHOTS (real project figures from report/figures/)
# ============================================================================
snapshot_files = [
    ("sample_data.png", "Multimodal MRI Sample"),
    ("training_curves.png", "Training Behaviour"),
    ("overlay.png", "Segmentation Overlay"),
]
available_snapshots = [(f, cap) for f, cap in snapshot_files if (FIGURES_DIR / f).exists()]
if available_snapshots:
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">Research Snapshot</div>
        <div class="section-sub">Figures from the accompanying research report.</div>
        """,
        unsafe_allow_html=True,
    )
    snap_cols = st.columns(len(available_snapshots))
    for col, (fname, caption) in zip(snap_cols, available_snapshots):
        b64 = _b64(FIGURES_DIR / fname)
        with col:
            st.markdown(
                f"<div class='snapshot-card'><img src='data:image/png;base64,{b64}'/>"
                f"<div class='cap'>{caption}</div></div>",
                unsafe_allow_html=True,
            )

# ============================================================================
# 11. FOOTER
# ============================================================================
st.markdown("<hr class='thin'/>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="app-footer">
        <div class="line1">Research demonstrator • Not for clinical diagnosis</div>
        <div>3D U-Net • BraTS 2020 • Multimodal MRI</div>
        <div style="margin-top:6px">
            <a href="{GITHUB_URL}" target="_blank">GitHub</a> ·
            <a href="{REPORT_URL}" target="_blank">Full report (PDF)</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
