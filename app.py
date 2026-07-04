import io
import re
import threading
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(
    page_title="AbsensiYOLO Dashboard",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "AbsensiYOLO"
APP_SUB = "AI ownership detector for photo, batch upload, video, webcam, checklist, and export log."
DEFAULT_WEIGHTS = "weights/best_items.pt"

if "log_rows" not in st.session_state:
    st.session_state["log_rows"] = []
if "avail_counts" not in st.session_state:
    st.session_state["avail_counts"] = {}
if "last_action" not in st.session_state:
    st.session_state["last_action"] = "Ready"


# =========================
# DATA + MODEL HELPERS
# =========================
def list_weight_files() -> List[str]:
    wdir = Path("weights")
    if not wdir.exists():
        return []
    return sorted(str(p).replace("\\", "/") for p in wdir.glob("*.pt"))


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> YOLO:
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {p.resolve()}")
    return YOLO(str(p))


def names_list(model: YOLO) -> List[str]:
    return [model.names[i] for i in sorted(model.names.keys())]


def to_keep_ids(model: YOLO, keep_names: List[str]) -> Optional[List[int]]:
    if not keep_names:
        return None
    inv = {v: k for k, v in model.names.items()}
    ids = [int(inv[nm]) for nm in keep_names if nm in inv]
    return ids if ids else None


def ensure_dir(path_str: str):
    Path(path_str).mkdir(parents=True, exist_ok=True)


def init_availability(all_labels: List[str]):
    for lb in all_labels:
        if lb not in st.session_state["avail_counts"]:
            st.session_state["avail_counts"][lb] = {"image": 0, "video": 0, "webcam": 0}


def reset_all(all_labels: List[str]):
    st.session_state["log_rows"] = []
    st.session_state["avail_counts"] = {
        lb: {"image": 0, "video": 0, "webcam": 0} for lb in all_labels
    }
    st.session_state["last_action"] = "Checklist and log reset"


def log_add(source: str, rows: List[Dict]):
    for row in rows:
        st.session_state["log_rows"].append({"source": source, **row})


def unique_labels(rows: List[Dict]) -> List[str]:
    return sorted({r["label"] for r in rows})


def best_label(rows: List[Dict]) -> Optional[Tuple[str, float]]:
    if not rows:
        return None
    best = max(rows, key=lambda row: row["conf"])
    return best["label"], float(best["conf"])


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-\. ]+", "_", stem)
    stem = stem.strip().replace(" ", "_")
    return stem[:120] if stem else "video"


def image_to_bytes(img_rgb: np.ndarray, fmt: str = "PNG") -> bytes:
    out = io.BytesIO()
    Image.fromarray(img_rgb).save(out, format=fmt)
    return out.getvalue()


def zip_bytes_from_files(files: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files:
            zf.writestr(name, data)
    return buf.getvalue()


def shrink_rgb(img_rgb: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img_rgb
    h, w = img_rgb.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img_rgb
    scale = max_side / float(side)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def infer_image_array(
    model: YOLO,
    img_rgb: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    keep_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        classes=keep_ids,
        verbose=False,
    )
    result = results[0]
    annotated_bgr = result.plot(conf=True, labels=True)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    rows: List[Dict] = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cid = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = [float(x) for x in box.xyxy[0].tolist()]
            rows.append(
                {
                    "label": model.names.get(cid, str(cid)),
                    "conf": round(score, 4),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                }
            )
    return annotated_rgb, rows


# =========================
# CHECKLIST + LOGIC
# =========================
def update_availability_image(labels: List[str]):
    for lb in set(labels):
        st.session_state["avail_counts"][lb]["image"] += 1


def update_availability_video(label: Optional[str], frame_idx: int, every_n_frames: int):
    if label and (every_n_frames <= 1 or frame_idx % every_n_frames == 0):
        st.session_state["avail_counts"][label]["video"] += 1


def update_availability_webcam(label: Optional[str], frame_idx: int, every_n_frames: int):
    if label and (every_n_frames <= 1 or frame_idx % every_n_frames == 0):
        st.session_state["avail_counts"][label]["webcam"] += 1


def availability_dataframe(all_labels: List[str], stream_threshold: int) -> pd.DataFrame:
    counts = st.session_state["avail_counts"]
    rows = []
    for label in all_labels:
        img_count = int(counts.get(label, {}).get("image", 0))
        vid_count = int(counts.get(label, {}).get("video", 0))
        cam_count = int(counts.get(label, {}).get("webcam", 0))
        ok_img = img_count >= 1
        ok_vid = vid_count >= stream_threshold
        ok_cam = cam_count >= stream_threshold
        total = img_count + vid_count + cam_count
        if ok_img or ok_vid or ok_cam:
            status = "OK"
            trigger = []
            if ok_img:
                trigger.append("foto")
            if ok_vid:
                trigger.append("video")
            if ok_cam:
                trigger.append("webcam")
            reason = " + ".join(trigger)
        elif total > 0:
            status = "Progress"
            reason = "belum memenuhi threshold"
        else:
            status = "Belum"
            reason = "-"
        rows.append(
            {
                "Label": label,
                "Foto": img_count,
                "Video": vid_count,
                "Webcam": cam_count,
                "Total": total,
                "Status": status,
                "Trigger": reason,
            }
        )
    return pd.DataFrame(rows)


def status_style(row):
    status = str(row["Status"])
    if status == "OK":
        return ["background-color:#0f3b24;color:#eafff2"] * len(row)
    if status == "Progress":
        return ["background-color:#4b2d12;color:#fff3d6"] * len(row)
    return ["background-color:#0d1118;color:#d7dae0"] * len(row)


def kpi_cards(rows: List[Dict]):
    c1, c2, c3, c4 = st.columns(4)
    if not rows:
        c1.metric("Detections", 0)
        c2.metric("Unique labels", 0)
        c3.metric("Avg conf", "-")
        c4.metric("Top label", "-")
        return
    df = pd.DataFrame(rows)
    c1.metric("Detections", len(df))
    c2.metric("Unique labels", int(df["label"].nunique()))
    c3.metric("Avg conf", f"{float(df['conf'].mean()):.2f}")
    c4.metric("Top label", df["label"].value_counts().index[0])


# =========================
# WEBRTC PROCESSOR
# =========================
class YOLOWebRTCProcessor(VideoProcessorBase):
    def __init__(
        self,
        model: YOLO,
        model_names: Dict[int, str],
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        keep_ids: Optional[List[int]],
        stable_n: int,
        every_n_webcam: int,
        skip_frames: int,
    ):
        self.model = model
        self.model_names = model_names
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.keep_ids = keep_ids
        self.stable_n = stable_n
        self.every_n_webcam = every_n_webcam
        self.skip_frames = max(1, int(skip_frames))
        self.lock = threading.Lock()
        self.frame_idx = 0
        self.hist: List[Optional[str]] = []
        self.last_best: Optional[Tuple[str, float]] = None
        self.last_stable: Optional[str] = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_idx += 1
        if self.frame_idx % self.skip_frames != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        results = self.model.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            classes=self.keep_ids,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot(conf=True, labels=True)

        rows: List[Dict] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cid = int(box.cls.item())
                score = float(box.conf.item())
                rows.append({"label": self.model_names.get(cid, str(cid)), "conf": score})

        best = best_label(rows)
        label = best[0] if best else None
        self.hist.append(label)
        if len(self.hist) > self.stable_n:
            self.hist = self.hist[-self.stable_n:]
        series = [x for x in self.hist if x is not None]
        stable = max(set(series), key=series.count) if series else None

        with self.lock:
            self.last_best = best
            self.last_stable = stable

        if stable:
            cv2.putText(
                annotated,
                f"STABLE: {stable}",
                (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (80, 255, 140),
                2,
                cv2.LINE_AA,
            )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# =========================
# PREMIUM UI STYLE
# =========================
st.markdown(
    """
    <style>
      :root {
        --bg:#07080d; --card:#111722; --card2:#0c111a; --line:rgba(255,255,255,.12);
        --text:#f7f2e8; --muted:#9da3ad; --red:#ff4f58; --blue:#65b7ff; --green:#6ff2a5;
      }
      .stApp {
        background:
          radial-gradient(circle at 15% 5%, rgba(255,79,88,.18), transparent 28%),
          radial-gradient(circle at 75% 0%, rgba(101,183,255,.16), transparent 30%),
          radial-gradient(circle at 86% 90%, rgba(111,242,165,.12), transparent 30%), var(--bg);
        color: var(--text);
      }
      .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1500px; }
      [data-testid="stSidebar"] { background: linear-gradient(180deg, #171a23, #10131b); border-right: 1px solid var(--line); }
      [data-testid="stSidebar"] * { font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
      .hero-card, .glass-card {
        border: 1px solid var(--line); border-radius: 28px;
        background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.025));
        box-shadow: 0 22px 70px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.08);
        backdrop-filter: blur(16px);
      }
      .hero-card { padding: 24px 26px; margin-bottom: 18px; position: relative; overflow: hidden; }
      .hero-card:before { content:""; position:absolute; inset:auto 24px 0 24px; height:2px; background:linear-gradient(90deg,var(--red),var(--blue),var(--green)); }
      .eyebrow { color: var(--red); font-size: 11px; font-weight: 900; letter-spacing: .22em; text-transform: uppercase; }
      .hero-title { font-size: clamp(38px, 5vw, 78px); line-height:.86; font-weight: 950; letter-spacing:-.07em; margin: 8px 0 12px; }
      .hero-sub { color: var(--muted); font-size: 15px; line-height: 1.45; max-width: 860px; }
      .pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
      .pill { border:1px solid var(--line); border-radius:999px; padding:8px 12px; background:rgba(255,255,255,.06); font-size:12px; font-weight:800; color:#d9dde5; }
      .section-card { border:1px solid var(--line); border-radius:24px; background:rgba(255,255,255,.035); padding:18px; margin-bottom:16px; }
      .section-title { font-size:26px; line-height:1; letter-spacing:-.04em; font-weight:950; margin: 0 0 8px; }
      .hint { color: var(--muted); font-size: 13px; margin-bottom: 14px; }
      div[data-testid="stMetric"] { border:1px solid var(--line); border-radius:20px; background:rgba(255,255,255,.055); padding:14px; }
      div[data-testid="stMetric"] label { color: var(--muted) !important; }
      .stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: 1px solid var(--line); }
      .stTabs [data-baseweb="tab"] { border-radius: 999px 999px 0 0; font-weight: 800; padding: 13px 18px; }
      .stTabs [aria-selected="true"] { color: #fff; border-bottom: 2px solid var(--red); }
      .stButton > button, .stDownloadButton > button {
        border-radius: 999px; border:1px solid rgba(255,255,255,.16); background: linear-gradient(135deg, #ff4f58, #d73d50); color:white; font-weight:900;
      }
      .stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--blue); transform: translateY(-1px); }
      [data-testid="stFileUploader"] { border:1px dashed rgba(255,255,255,.20); border-radius:22px; padding:8px; background:rgba(255,255,255,.04); }
      [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:18px; overflow:hidden; }
      .status-ok { color: var(--green); font-weight: 900; }
      .small-note { color: var(--muted); font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-card">
      <div class="eyebrow">Realtime YOLO Attendance Asset Control</div>
      <div class="hero-title">{APP_TITLE}</div>
      <div class="hero-sub">{APP_SUB}</div>
      <div class="pill-row">
        <span class="pill">Photo detection</span>
        <span class="pill">Batch ZIP</span>
        <span class="pill">Video processing</span>
        <span class="pill">WebRTC live camera</span>
        <span class="pill">Checklist CSV</span>
        <span class="pill">Detection log</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# SIDEBAR SETTINGS
# =========================
st.sidebar.markdown("## Settings")
weights_list = list_weight_files() or [DEFAULT_WEIGHTS]
default_idx = weights_list.index(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS in weights_list else 0
weights_path = st.sidebar.selectbox("Model weights (.pt)", weights_list, index=default_idx)

try:
    with st.spinner("Loading YOLO model..."):
        model = load_model(weights_path)
    classes = names_list(model)
    init_availability(classes)
    st.sidebar.success("Model loaded")
except Exception as exc:
    st.sidebar.error(str(exc))
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### Detection")
conf = st.sidebar.slider("Confidence", 0.05, 0.99, 0.60, 0.01)
iou = st.sidebar.slider("IoU", 0.10, 0.95, 0.50, 0.01)
imgsz = st.sidebar.selectbox("Image size", [320, 480, 640, 768], index=2)
max_det = st.sidebar.slider("Max detections", 1, 200, 50, 1)
keep_classes = st.sidebar.multiselect("Keep only classes", options=classes, default=[])
keep_ids = to_keep_ids(model, keep_classes)

st.sidebar.markdown("---")
st.sidebar.markdown("### Checklist Rules")
stream_threshold = st.sidebar.slider("Green threshold video/webcam", 1, 30, 3, 1)
every_n_video = st.sidebar.slider("Video counter every N frames", 1, 30, 5, 1)
every_n_webcam = st.sidebar.slider("Webcam counter every N frames", 1, 30, 5, 1)
stable_n = st.sidebar.slider("Stable vote frames", 1, 21, 9, 1)
skip_frames = st.sidebar.slider("Webcam process every N frames", 1, 10, 2, 1)
max_side = st.sidebar.selectbox("Max photo side", [0, 1280, 1024, 800], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Export")
local_save = st.sidebar.checkbox("Save outputs locally", value=False)
outdir_local = st.sidebar.text_input("Image output folder", value="outputs/pred_images")
outdir_video = st.sidebar.text_input("Video output folder", value="outputs/videos")

if st.sidebar.button("Reset checklist + log", use_container_width=True):
    reset_all(classes)
    st.rerun()

with st.sidebar.expander("Model class names"):
    st.code(str(model.names), language="python")


# =========================
# MAIN LAYOUT
# =========================
log_rows = st.session_state["log_rows"]
avail_df = availability_dataframe(classes, stream_threshold)
ok_count = int((avail_df["Status"] == "OK").sum())
progress_count = int((avail_df["Status"] == "Progress").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Labels OK", f"{ok_count}/{len(avail_df)}")
m2.metric("In progress", progress_count)
m3.metric("Total log", len(log_rows))
m4.metric("Last action", st.session_state["last_action"])

left, right = st.columns([3.1, 1.15], gap="large")

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Checklist</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hint">Green when photo is detected once, or video/webcam reaches the selected threshold.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        avail_df.style.apply(status_style, axis=1),
        use_container_width=True,
        height=520,
    )
    st.download_button(
        "Download checklist CSV",
        data=avail_df.to_csv(index=False).encode("utf-8"),
        file_name="checklist_ketersediaan.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with left:
    tabs = st.tabs(["Photo", "Batch / ZIP", "Video", "Webcam LIVE", "Log & Export"])

    with tabs[0]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Photo Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="hint">Upload one image, run detection, then download the annotated result.</div>', unsafe_allow_html=True)
        up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        if up is not None:
            img_rgb = np.array(Image.open(up).convert("RGB"))
            img_rgb = shrink_rgb(img_rgb, int(max_side))
            c1, c2 = st.columns(2)
            c1.image(Image.fromarray(img_rgb), caption="Input", use_container_width=True)
            if st.button("Detect this photo", type="primary", use_container_width=True):
                annotated_rgb, rows = infer_image_array(model, img_rgb, conf, iou, imgsz, max_det, keep_ids)
                c2.image(annotated_rgb, caption="Output", use_container_width=True)
                labels = unique_labels(rows)
                st.session_state["last_action"] = f"Photo: {up.name}"
                if rows:
                    st.success(f"Detected: {', '.join(labels)}")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    log_add(source=f"image:{up.name}", rows=rows)
                    update_availability_image(labels)
                    if local_save:
                        ensure_dir(outdir_local)
                        Image.fromarray(annotated_rgb).save(Path(outdir_local) / f"pred_{Path(up.name).stem}.png")
                    st.download_button(
                        "Download PNG",
                        data=image_to_bytes(annotated_rgb, "PNG"),
                        file_name=f"pred_{Path(up.name).stem}.png",
                        mime="image/png",
                    )
                else:
                    st.info("No detection above threshold.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Batch Photo / ZIP</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        ups = col_a.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="multi_img")
        zip_up = col_b.file_uploader("Upload ZIP images", type=["zip"], key="zip_img")
        images: List[Tuple[str, np.ndarray]] = []
        if zip_up is not None:
            with zipfile.ZipFile(io.BytesIO(zip_up.getvalue()), "r") as zf:
                for name in zf.namelist():
                    if name.lower().endswith((".jpg", ".jpeg", ".png")):
                        arr = np.array(Image.open(io.BytesIO(zf.read(name))).convert("RGB"))
                        images.append((Path(name).name, shrink_rgb(arr, int(max_side))))
        if ups:
            for file in ups:
                arr = np.array(Image.open(file).convert("RGB"))
                images.append((file.name, shrink_rgb(arr, int(max_side))))
        if images:
            st.info(f"Ready to process {len(images)} images.")
            if st.button("Process all images", type="primary", use_container_width=True):
                progress = st.progress(0)
                out_files: List[Tuple[str, bytes]] = []
                all_rows: List[Dict] = []
                if local_save:
                    ensure_dir(outdir_local)
                for i, (name, arr) in enumerate(images, start=1):
                    annotated, rows = infer_image_array(model, arr, conf, iou, imgsz, max_det, keep_ids)
                    out_name = f"pred_{Path(name).stem}.png"
                    out_files.append((out_name, image_to_bytes(annotated)))
                    labels = unique_labels(rows)
                    if labels:
                        update_availability_image(labels)
                    if rows:
                        log_add(source=f"batch:{name}", rows=rows)
                        all_rows.extend([{"source": f"batch:{name}", **row} for row in rows])
                    if local_save:
                        Image.fromarray(annotated).save(Path(outdir_local) / out_name)
                    progress.progress(i / len(images))
                st.session_state["last_action"] = f"Batch: {len(images)} images"
                st.success("Batch processing complete.")
                st.download_button("Download all results ZIP", data=zip_bytes_from_files(out_files), file_name="pred_images.zip", mime="application/zip")
                if all_rows:
                    df = pd.DataFrame(all_rows)
                    st.dataframe(df.groupby("label")["conf"].agg(["count", "mean"]).sort_values("count", ascending=False), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Video Detection</div>', unsafe_allow_html=True)
        vup = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv", "mpeg4"], accept_multiple_files=False)
        out_format = st.radio("Output format", ["AVI (XVID)", "MP4 (mp4v)"], horizontal=True)
        if vup is not None:
            st.video(vup)
            if st.button("Process video", type="primary", use_container_width=True):
                ensure_dir(outdir_video)
                tmpdir = Path(outdir_video) / "_tmp_upload"
                tmpdir.mkdir(parents=True, exist_ok=True)
                in_path = tmpdir / vup.name
                in_path.write_bytes(vup.getvalue())
                cap = cv2.VideoCapture(str(in_path))
                if not cap.isOpened():
                    st.error("Failed to open input video.")
                    st.stop()
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                stem = safe_stem(vup.name)
                if out_format.startswith("AVI"):
                    out_path = Path(outdir_video) / f"pred_{stem}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                else:
                    out_path = Path(outdir_video) / f"pred_{stem}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
                if not writer.isOpened():
                    cap.release()
                    st.error("VideoWriter failed. Try AVI output.")
                    st.stop()
                progress = st.progress(0)
                status = st.empty()
                hist: List[Optional[str]] = []
                frame_idx = 0
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame_idx += 1
                        results = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=keep_ids, verbose=False)
                        result = results[0]
                        annotated_bgr = result.plot(conf=True, labels=True)
                        rows: List[Dict] = []
                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                cid = int(box.cls.item())
                                score = float(box.conf.item())
                                rows.append({"label": model.names.get(cid, str(cid)), "conf": round(score, 4)})
                        best = best_label(rows)
                        label = best[0] if best else None
                        hist.append(label)
                        if len(hist) > stable_n:
                            hist = hist[-stable_n:]
                        series = [x for x in hist if x is not None]
                        stable = max(set(series), key=series.count) if series else None
                        update_availability_video(stable, frame_idx, every_n_video)
                        if stable:
                            cv2.putText(annotated_bgr, f"STABLE: {stable}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 140), 2, cv2.LINE_AA)
                        writer.write(annotated_bgr)
                        if best:
                            st.session_state["log_rows"].append({"source": f"video:{vup.name}@{frame_idx}", "label": best[0], "conf": round(best[1], 4), "x1": None, "y1": None, "x2": None, "y2": None})
                        if total > 0:
                            progress.progress(min(frame_idx / total, 1.0))
                            status.write(f"Frame {frame_idx}/{total}")
                        else:
                            status.write(f"Frame {frame_idx}")
                finally:
                    cap.release()
                    writer.release()
                st.session_state["last_action"] = f"Video: {vup.name}"
                st.success(f"Done: {out_path}")
                st.download_button("Download processed video", data=Path(out_path).read_bytes(), file_name=out_path.name, mime="video/mp4" if out_path.suffix.lower() == ".mp4" else "video/x-msvideo")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Webcam LIVE</div>', unsafe_allow_html=True)
        st.caption("Click START, allow camera permission, then sync status into checklist/log.")
        rtc_conf = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

        def processor_factory():
            return YOLOWebRTCProcessor(model, model.names, conf, iou, imgsz, max_det, keep_ids, stable_n, every_n_webcam, skip_frames)

        webrtc_ctx = webrtc_streamer(
            key="yolo-live",
            video_processor_factory=processor_factory,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=rtc_conf,
            async_processing=True,
        )
        col1, col2, col3 = st.columns(3)
        if st.button("Sync webcam status", use_container_width=True):
            vp = webrtc_ctx.video_processor if webrtc_ctx else None
            if vp is None:
                st.warning("Webcam is not ready yet.")
            else:
                with vp.lock:
                    last_best = vp.last_best
                    last_stable = vp.last_stable
                    frame_idx = vp.frame_idx
                if last_best:
                    st.session_state["log_rows"].append({"source": f"webcam@{frame_idx}", "label": last_best[0], "conf": round(float(last_best[1]), 4), "x1": None, "y1": None, "x2": None, "y2": None})
                update_availability_webcam(last_stable, frame_idx, every_n_webcam)
                st.session_state["last_action"] = f"Webcam stable: {last_stable or '-'}"
                st.success(f"Synced. Stable={last_stable or '-'} | Best={last_best[0] if last_best else '-'}")
        vp2 = webrtc_ctx.video_processor if webrtc_ctx else None
        if vp2 is not None:
            with vp2.lock:
                lb = vp2.last_best
                stbl = vp2.last_stable
                fidx = vp2.frame_idx
            col1.metric("Frame processed", fidx)
            col2.metric("Stable label", stbl if stbl else "-")
            col3.metric("Best conf", f"{lb[1]:.2f}" if lb else "-")
        st.info("If the webcam feels laggy, increase process every N frames or reduce image size.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Detection Log & Export</div>', unsafe_allow_html=True)
        rows = st.session_state["log_rows"]
        kpi_cards(rows)
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=420)
            st.markdown("### Label summary")
            st.dataframe(df.groupby("label")["conf"].agg(["count", "mean"]).sort_values("count", ascending=False), use_container_width=True)
            st.download_button("Download log CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="detections_log.csv", mime="text/csv")
            if st.button("Clear log"):
                st.session_state["log_rows"] = []
                st.session_state["last_action"] = "Log cleared"
                st.rerun()
        else:
            st.info("Log is empty. Run detection from photo, video, or webcam.")
        st.markdown("</div>", unsafe_allow_html=True)
