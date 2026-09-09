"""
Streamlit app: upload a brain MRI (four co-registered modalities) and get an
automatic glioblastoma segmentation — tumor location overlaid on the scan, or a
"no tumor detected" message.

Run from the repo root:
    streamlit run app/streamlit_app.py

The trained checkpoint path defaults to the bundled fp16 weights
(app/weights/), or BRATS_CHECKPOINT (env); override it in the sidebar.
"""

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

EXAMPLE_META = {
    "BraTS20_Training_341": ("🔹", "Small tumor"),
    "BraTS20_Training_049": ("🔸", "Medium tumor"),
    "BraTS20_Training_001": ("🔴", "Large tumor"),
}
MODALITY_ICONS = {"t1": "🧠", "t1ce": "💉", "t2": "🌊", "flair": "✨"}

st.set_page_config(page_title="Glioblastoma Tumor Detector", page_icon="🧠", layout="wide")

# A little custom CSS — sized to widget internals that don't hardcode colors,
# so it stays correct in both light and dark Streamlit themes.
st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] { height: 2.9em; font-size: 1.05em; font-weight: 600; }
    .legend-row { margin-top: 0.5rem; }
    .legend-chip { display: inline-flex; align-items: center; gap: 6px; margin-right: 20px; font-size: 0.9em; }
    .legend-swatch { width: 13px; height: 13px; border-radius: 4px; display: inline-block; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _swatch(rgb: tuple[int, int, int]) -> str:
    return f"<span class='legend-swatch' style='background:rgb{rgb}'></span>"


def _legend_html() -> str:
    chips = "".join(
        f"<span class='legend-chip'>{_swatch(inf.CLASS_COLORS[c])} {inf.CLASS_NAMES[c]}</span>"
        for c in (1, 2, 3)
    )
    return f"<div class='legend-row'>{chips}</div>"


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


# ============================== Header ==============================
title_col, badge_col = st.columns([3, 2], vertical_alignment="center")
with title_col:
    st.title("🧠 Glioblastoma Tumor Detector")
    st.caption(
        "Automatic detection and segmentation of glioblastoma sub-regions from "
        "multi-modal brain MRI, using a 3D U-Net trained on BraTS 2020."
    )
with badge_col:
    b1, b2, b3 = st.columns(3)
    with b1:
        st.badge("3D U-Net", icon="🧩", color="blue")
    with b2:
        st.badge("BraTS 2020 · HGG", icon="🎗️", color="violet")
    with b3:
        st.badge("Val. Dice 0.71", icon="📈", color="green")

st.divider()

# ============================== Sidebar ==============================
with st.sidebar:
    st.header("About")
    st.markdown(
        "This app runs a 3D U-Net that segments glioblastoma sub-regions "
        "(necrotic core, edema, enhancing tumor) from four co-registered MRI "
        "modalities (T1, T1Gd, T2, FLAIR).\n\n"
        f"📄 [Full write-up (PDF)]({REPORT_URL})  \n"
        f"💻 [Source on GitHub]({GITHUB_URL})"
    )
    st.divider()
    with st.expander("⚙️ Advanced settings"):
        ckpt = st.text_input("Checkpoint path", value=DEFAULT_CKPT)
        device_pref = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
        data_root = st.text_input("Local BraTS data root (optional)", value=DEFAULT_DATA_ROOT)

if not Path(ckpt).exists():
    st.error(
        f"Model checkpoint not found at:\n\n`{ckpt}`\n\n"
        "Set the correct path in the sidebar under **Advanced settings** (or the "
        "`BRATS_CHECKPOINT` environment variable). This is the trained `*_best.pt` file."
    )
    st.stop()

model, epoch, device = _get_model(ckpt, device_pref)
with st.sidebar:
    st.success(f"Model ready — epoch {epoch} · {device.type.upper()}")

# ============================== Input ==============================
st.subheader("1. Choose a scan")

volumes = None          # list of 4 arrays [t1, t1ce, t2, flair]
gt = None                # optional ground-truth label volume
source_label = ""

mode = st.segmented_control(
    "Input source",
    ["✨ Try an example", "📤 Upload your scan"],
    default="✨ Try an example",
    label_visibility="collapsed",
)

if mode == "📤 Upload your scan":
    with st.container(border=True):
        st.markdown(
            "Upload the **four co-registered MRI modalities** (BraTS-style, "
            "skull-stripped, 1 mm isotropic). A single scan cannot be segmented "
            "reliably — the model needs all four."
        )
        cols = st.columns(4)
        uploads = {}
        for col, mod in zip(cols, inf.MODALITIES):
            with col:
                st.markdown(f"{MODALITY_ICONS.get(mod, '')} **{inf.MODALITY_LABELS[mod]}**")
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
    with st.container(border=True):
        bundled = inf.list_bundled_examples()
        if not bundled:
            st.error("No bundled examples found under app/examples/.")
        else:
            pid = st.pills(
                "Example subject",
                list(bundled.keys()),
                format_func=lambda p: f"{EXAMPLE_META.get(p, ('', p))[0]} {EXAMPLE_META.get(p, ('', p))[1]}",
                default=list(bundled.keys())[0],
                required=True,
            )
            show_gt = st.toggle("Show ground truth for comparison", value=True)
            if pid:
                volumes, gt_full = inf.load_bundled_example(pid)
                _, label = EXAMPLE_META.get(pid, ("", pid))
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

# ============================== Run ==============================
st.subheader("2. Run the analysis")
analyze = st.button(
    "🔍 Analyze scan", type="primary", use_container_width=True, disabled=(volumes is None)
)
if volumes is None:
    st.caption("Select an example or upload a scan above to enable analysis.")

if volumes is not None and analyze:
    with st.spinner("Running 3D segmentation… this can take up to a minute on CPU."):
        image = inf.preprocess(volumes)             # (4, D, H, W), z-scored + cropped
        pred = inf.predict(model, image, device)    # (D, H, W)
        stats = inf.summarize(pred)

    st.divider()
    st.subheader(f"Result — {source_label}")

    with st.container(border=True):
        if not stats["tumor_present"]:
            st.markdown("### ✅ No tumor detected")
            st.caption("The model found no significant tumor tissue in this scan.")
        else:
            st.markdown(f"### ⚠️ Tumor detected — {stats['total_ml']} mL total")
            c1, c2, c3 = st.columns(3)
            for col, cls in zip((c1, c2, c3), (3, 2, 1)):  # ET, ED, NCR
                with col:
                    st.markdown(f"{_swatch(inf.CLASS_COLORS[cls])} **{inf.CLASS_NAMES[cls]}**", unsafe_allow_html=True)
                    st.metric(label="", value=f"{stats['per_class_ml'][cls]} mL", label_visibility="collapsed")

    with st.container(border=True):
        st.markdown("#### 🔬 Slice viewer")
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
        s = st.slider("Axial slice", 0, pred.shape[0] - 1, default_slice)

        if gt_crop is not None:
            v1, v2, v3 = st.columns(3)
            v1.image(inf.make_overlay(flair[s], np.zeros_like(pred[s])), caption="MRI (FLAIR)", use_container_width=True)
            v2.image(inf.make_overlay(flair[s], gt_crop[s]), caption="Ground truth", use_container_width=True)
            v3.image(inf.make_overlay(flair[s], pred[s]), caption="Prediction", use_container_width=True)
        else:
            v1, v2 = st.columns(2)
            v1.image(inf.make_overlay(flair[s], np.zeros_like(pred[s])), caption="MRI (FLAIR)", use_container_width=True)
            v2.image(inf.make_overlay(flair[s], pred[s]), caption="Predicted tumor", use_container_width=True)

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

st.divider()
st.caption(
    f"Research demo only — not for clinical use. [GitHub]({GITHUB_URL}) · [Full report (PDF)]({REPORT_URL})"
)
