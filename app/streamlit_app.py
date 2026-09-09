"""
Streamlit app: upload a brain MRI (four co-registered modalities) and get an
automatic glioblastoma segmentation — tumour location overlaid on the scan, or a
"no tumour detected" message.

Run from the repo root:
    streamlit run app/streamlit_app.py

The trained checkpoint path defaults to BRATS_CHECKPOINT (env) or the local
U:\\brats_outputs path; override it in the sidebar.
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

st.set_page_config(page_title="Glioblastoma Tumour Detector", page_icon="🧠", layout="wide")


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


# ----------------------------- Sidebar -----------------------------
st.sidebar.header("Settings")
ckpt = st.sidebar.text_input("Checkpoint path", value=DEFAULT_CKPT)
device_pref = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
data_root = st.sidebar.text_input("BraTS data root (for examples)", value=DEFAULT_DATA_ROOT)

st.title("🧠 Glioblastoma Tumour Detection & Segmentation")
st.caption(
    "Upload a brain MRI (four co-registered modalities) or try an example. "
    "A 3D U-Net segments the tumour sub-regions and reports whether a tumour is present."
)

if not Path(ckpt).exists():
    st.error(
        f"Model checkpoint not found at:\n\n`{ckpt}`\n\n"
        "Set the correct path in the sidebar (or the `BRATS_CHECKPOINT` environment "
        "variable). This is the trained `*_best.pt` file."
    )
    st.stop()

model, epoch, device = _get_model(ckpt, device_pref)
st.sidebar.success(f"Model loaded (epoch {epoch}) · device: {device.type}")

# ----------------------------- Input -----------------------------
# "Try an example" uses subjects bundled with the app (app/examples/), so it
# always works, including on a cloud deployment with no access to the full
# local BraTS dataset.
EXAMPLE_LABELS = {
    "BraTS20_Training_341": "Small tumor",
    "BraTS20_Training_049": "Medium tumor",
    "BraTS20_Training_001": "Large tumor",
}

volumes = None          # list of 4 arrays [t1, t1ce, t2, flair]
gt = None               # optional ground-truth label volume
source_label = ""

mode = st.radio(
    "Input",
    ["Try an example", "Upload MRI (4 modalities)"],
    horizontal=True,
)

if mode == "Upload MRI (4 modalities)":
    st.info(
        "Upload the four co-registered, skull-stripped modalities (BraTS-style, "
        "1 mm isotropic). A single scan cannot be segmented reliably — the model needs all four."
    )
    cols = st.columns(4)
    uploads = {}
    for col, mod in zip(cols, inf.MODALITIES):
        with col:
            uploads[mod] = st.file_uploader(inf.MODALITY_LABELS[mod], type=["nii", "gz"], key=mod)
    if all(uploads[m] is not None for m in inf.MODALITIES):
        try:
            volumes = [inf.load_nifti(_save_upload(uploads[m])) for m in inf.MODALITIES]
            source_label = "uploaded scan"
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the uploaded files: {e}")

else:  # Try an example — bundled with the app
    bundled = inf.list_bundled_examples()
    if not bundled:
        st.error("No bundled examples found under app/examples/.")
    else:
        pid = st.selectbox(
            "Example subject",
            list(bundled.keys()),
            format_func=lambda p: f"{EXAMPLE_LABELS.get(p, p)} ({p})",
        )
        show_gt = st.checkbox("Show ground truth for comparison", value=True)
        if pid:
            volumes, gt_full = inf.load_bundled_example(pid)
            source_label = f"{EXAMPLE_LABELS.get(pid, pid)} — {pid}"
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

# ----------------------------- Run + results -----------------------------
if volumes is not None and st.button("Analyze scan", type="primary"):
    with st.spinner("Running segmentation..."):
        image = inf.preprocess(volumes)             # (4, D, H, W), z-scored + cropped
        pred = inf.predict(model, image, device)    # (D, H, W)
        stats = inf.summarize(pred)

    st.subheader(f"Result — {source_label}")
    if not stats["tumor_present"]:
        st.success("✅ No tumour detected.")
    else:
        st.error(f"⚠️ Tumour detected — estimated volume {stats['total_ml']} mL "
                 f"({stats['total_voxels']:,} voxels).")
        c1, c2, c3 = st.columns(3)
        for col, cls in zip((c1, c2, c3), (3, 2, 1)):  # ET, ED, NCR
            col.metric(inf.CLASS_NAMES[cls], f"{stats['per_class_ml'][cls]} mL")

    # Slice viewer
    flair = image[inf.MODALITIES.index("flair")]    # (D, H, W), z-scored (fine for display)
    default_slice = inf.best_tumor_slice(pred)
    s = st.slider("Axial slice", 0, pred.shape[0] - 1, default_slice)

    if gt is not None:
        gt_crop = inf._center_crop(gt[np.newaxis].astype(np.float32))[0].astype(np.int64)
        v1, v2, v3 = st.columns(3)
        v1.image(inf.make_overlay(flair[s], np.zeros_like(pred[s])), caption="MRI (FLAIR)", use_container_width=True)
        v2.image(inf.make_overlay(flair[s], gt_crop[s]), caption="Ground truth", use_container_width=True)
        v3.image(inf.make_overlay(flair[s], pred[s]), caption="Prediction", use_container_width=True)
    else:
        v1, v2 = st.columns(2)
        v1.image(inf.make_overlay(flair[s], np.zeros_like(pred[s])), caption="MRI (FLAIR)", use_container_width=True)
        v2.image(inf.make_overlay(flair[s], pred[s]), caption="Predicted tumour", use_container_width=True)

    st.caption("Legend — 🟥 necrotic/non-enhancing core · 🟩 edema · 🟦 enhancing tumour. "
               "Research demo only; not for clinical use.")
