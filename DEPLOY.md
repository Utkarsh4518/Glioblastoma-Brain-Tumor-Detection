# Deploying the web app

The Streamlit app (`app/streamlit_app.py`) is self-contained: the model weights are
bundled in `app/weights/brats_hgg_unet_fp16.pt` (~38 MB), and `requirements.txt` is a
slim runtime spec. So it deploys straight from this GitHub repo with no external weight
hosting and no secrets.

## Streamlit Community Cloud (free, recommended)

1. Push the repo to GitHub (already done).
2. Go to **https://share.streamlit.io** and sign in with your GitHub account.
3. Click **Create app → Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `Utkarsh4518/Glioblastoma-Brain-Tumor-Detection`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
5. Click **Deploy**. The first build takes a few minutes (it installs `requirements.txt`
   and clones the bundled weights).
6. You get a public URL like `https://<your-app>.streamlit.app` — it works from any
   device, which is what you want.

### What to expect on the free tier
- **CPU only.** A 3D segmentation of one scan takes roughly 1–3 minutes (there is a
  spinner). This is a limitation of the free CPU tier, not the model.
- **Upload mode only.** The "Try an example" option needs the local BraTS dataset, which
  is not on the cloud, so the app defaults to the **Upload MRI** tab there. Users upload
  the four co-registered modalities (T1, T1Gd, T2, FLAIR) as `.nii`/`.nii.gz`.
- If a build hits the memory limit, lower the inference patch size (`ROI` in
  `app/inference.py`, e.g. `(96, 96, 96)`) and redeploy.

## Alternative: Hugging Face Spaces

1. Create a free account at https://huggingface.co and a new **Space** (SDK: *Streamlit*).
2. Push this repo's contents to the Space (the bundled weights come along).
3. Set the Space's app file to `app/streamlit_app.py`.
Spaces' free CPU tier has more RAM than Streamlit Cloud but is otherwise similar.

## Notes
- To point the app at a different checkpoint, set the `BRATS_CHECKPOINT` environment
  variable (or the sidebar field) — otherwise it uses the bundled weights.
- The app is a research demo and is not for clinical use.
