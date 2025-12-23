import io
import re
import time
import zipfile
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# OpenCV (headless OK)
import cv2

from ultralytics import YOLO

# WebRTC live webcam
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Budiyolo Dashboard", layout="wide")

APP_TITLE = "🧠 Budiyolo — Items Ownership Detector"
APP_SUB = "Foto • Batch • Video • Webcam LIVE (WebRTC) • Checklist Ketersediaan • Log & Export"
DEFAULT_WEIGHTS = "weights/best_items.pt"


# =========================
# SESSION STATE INIT
# =========================
if "log_rows" not in st.session_state:
    st.session_state["log_rows"] = []

# label -> {"image":0,"video":0,"webcam":0}
if "avail_counts" not in st.session_state:
    st.session_state["avail_counts"] = {}


# =========================
# HELPERS
# =========================
def list_weight_files() -> List[str]:
    wdir = Path("weights")
    if not wdir.exists():
        return []
    pts = sorted([str(p).replace("\\", "/") for p in wdir.glob("*.pt")])
    return pts


@st.cache_resource
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
    st.session_state["avail_counts"] = {lb: {"image": 0, "video": 0, "webcam": 0} for lb in all_labels}


def log_add(source: str, rows: List[Dict]):
    for r in rows:
        st.session_state["log_rows"].append({"source": source, **r})


def unique_labels(rows: List[Dict]) -> List[str]:
    return sorted({r["label"] for r in rows})


def best_label(rows: List[Dict]) -> Optional[Tuple[str, float]]:
    if not rows:
        return None
    b = max(rows, key=lambda r: r["conf"])
    return b["label"], float(b["conf"])


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-\. ]+", "_", stem)
    stem = stem.strip().replace(" ", "_")
    return stem[:120] if stem else "video"


def image_to_bytes(img_rgb: np.ndarray, fmt: str = "PNG") -> bytes:
    im = Image.fromarray(img_rgb)
    out = io.BytesIO()
    im.save(out, format=fmt)
    return out.getvalue()


def zip_bytes_from_files(files: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, b in files:
            z.writestr(name, b)
    return buf.getvalue()


def shrink_rgb(img_rgb: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img_rgb
    h, w = img_rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_rgb
    scale = max_side / float(m)
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
    r = results[0]
    annotated_bgr = r.plot(conf=True, labels=True)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    rows: List[Dict] = []
    if r.boxes is not None and len(r.boxes) > 0:
        nm = model.names
        for b in r.boxes:
            cid = int(b.cls.item())
            score = float(b.conf.item())
            x1, y1, x2, y2 = [float(x) for x in b.xyxy[0].tolist()]
            rows.append(
                {
                    "label": nm.get(cid, str(cid)),
                    "conf": round(score, 4),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                }
            )
    return annotated_rgb, rows


# =========================
# CHECKLIST UPDATE
# =========================
def update_availability_image(labels: List[str]):
    for lb in set(labels):
        st.session_state["avail_counts"][lb]["image"] += 1


def update_availability_video(label: Optional[str], frame_idx: int, every_n_frames: int):
    if not label:
        return
    if every_n_frames <= 1:
        st.session_state["avail_counts"][label]["video"] += 1
        return
    if frame_idx % every_n_frames == 0:
        st.session_state["avail_counts"][label]["video"] += 1


def update_availability_webcam(label: Optional[str], frame_idx: int, every_n_frames: int):
    if not label:
        return
    if every_n_frames <= 1:
        st.session_state["avail_counts"][label]["webcam"] += 1
        return
    if frame_idx % every_n_frames == 0:
        st.session_state["avail_counts"][label]["webcam"] += 1


def render_availability_panel(all_labels: List[str], stream_threshold: int):
    counts = st.session_state["avail_counts"]
    rows = []
    for lb in all_labels:
        cimg = int(counts.get(lb, {}).get("image", 0))
        cvid = int(counts.get(lb, {}).get("video", 0))
        ccam = int(counts.get(lb, {}).get("webcam", 0))

        ok_img = cimg >= 1
        ok_vid = cvid >= stream_threshold
        ok_cam = ccam >= stream_threshold

        is_green = ok_img or ok_vid or ok_cam
        is_yellow = (cimg + cvid + ccam) > 0 and not is_green

        if is_green:
            status = "✅ OK"
            sebab = []
            if ok_img:
                sebab.append("foto≥1")
            if ok_vid:
                sebab.append(f"video≥{stream_threshold}")
            if ok_cam:
                sebab.append(f"webcam≥{stream_threshold}")
            trigger = " / ".join(sebab)
        elif is_yellow:
            status = "🟡 Terdeteksi"
            trigger = "belum memenuhi syarat hijau"
        else:
            status = "⚪ Belum"
            trigger = "-"

        rows.append(
            {
                "Label": lb,
                "Foto": cimg,
                "Video": cvid,
                "Webcam": ccam,
                "Status": status,
                "Trigger": trigger,
            }
        )

    df = pd.DataFrame(rows)

    def style_row(row):
        if str(row["Status"]).startswith("✅"):
            return ["background-color:#14532d;color:white"] * len(row)
        if str(row["Status"]).startswith("🟡"):
            return ["background-color:#7c2d12;color:white"] * len(row)
        return ["background-color:#111827;color:#e5e7eb"] * len(row)

    st.markdown("### ✅ Checklist Ketersediaan (Hari Ini)")
    st.caption("Hijau jika salah satu terpenuhi: foto≥1 ATAU video≥threshold ATAU webcam≥threshold (OR antar sumber).")
    st.dataframe(df.style.apply(style_row, axis=1), use_container_width=True, height=520)

    st.download_button(
        "⬇️ Download Checklist CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="checklist_ketersediaan.csv",
        mime="text/csv",
    )

    ok = df[df["Status"].str.startswith("✅")]
    st.write(f"**OK (Hijau):** {len(ok)} / {len(df)}")


def kpi_cards(rows: List[Dict]):
    c1, c2, c3, c4 = st.columns(4)
    if not rows:
        c1.metric("Total detections", 0)
        c2.metric("Unique labels", 0)
        c3.metric("Avg confidence", "-")
        c4.metric("Top label", "-")
        return
    df = pd.DataFrame(rows)
    total = len(df)
    uniq = int(df["label"].nunique())
    avgc = float(df["conf"].mean()) if total else 0.0
    top = df["label"].value_counts().index[0] if total else "-"
    c1.metric("Total detections", total)
    c2.metric("Unique labels", uniq)
    c3.metric("Avg confidence", f"{avgc:.2f}")
    c4.metric("Top label", top)


# =========================
# LIVE WEBCAM PROCESSOR (WEBRTC)
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

        # Skip frame untuk mempercepat
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
        r = results[0]
        annotated = r.plot(conf=True, labels=True)  # BGR

        # ambil best label
        rows: List[Dict] = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cid = int(b.cls.item())
                score = float(b.conf.item())
                rows.append({"label": self.model_names.get(cid, str(cid)), "conf": score})

        bl = best_label(rows)
        lbl = bl[0] if bl else None

        # stabilizer
        self.hist.append(lbl)
        if len(self.hist) > self.stable_n:
            self.hist = self.hist[-self.stable_n :]
        series = [x for x in self.hist if x is not None]
        stable = max(set(series), key=series.count) if series else None

        with self.lock:
            self.last_best = bl
            self.last_stable = stable

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# =========================
# UI STYLE
# =========================
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }
      .soft-card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
      }
      .title { font-size: 1.55rem; font-weight: 800; }
      .sub { opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="soft-card">
      <div class="title">{APP_TITLE}</div>
      <div class="sub">{APP_SUB}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# SIDEBAR SETTINGS
# =========================
st.sidebar.header("⚙️ Settings")

weights_list = list_weight_files()
if not weights_list:
    weights_list = [DEFAULT_WEIGHTS]
default_idx = weights_list.index(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS in weights_list else 0
weights_path = st.sidebar.selectbox("Model weights (.pt)", weights_list, index=default_idx)

try:
    model = load_model(weights_path)
    classes = names_list(model)
    init_availability(classes)
    st.sidebar.success("Model loaded ✅")
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

conf = st.sidebar.slider("Confidence", 0.05, 0.99, 0.60, 0.01)
iou = st.sidebar.slider("IoU", 0.10, 0.95, 0.50, 0.01)
imgsz = st.sidebar.selectbox("imgsz", [320, 480, 640, 768], index=2)
max_det = st.sidebar.slider("Max detections", 1, 200, 50, 1)

st.sidebar.markdown("---")
keep_classes = st.sidebar.multiselect("Keep only (opsional)", options=classes, default=[])
keep_ids = to_keep_ids(model, keep_classes)

st.sidebar.markdown("---")
st.sidebar.subheader("Checklist Rules")
stream_threshold = st.sidebar.slider("Threshold hijau (Video/Webcam)", 1, 30, 3, 1)
every_n_video = st.sidebar.slider("Video: tambah counter tiap N frame", 1, 30, 5, 1)
every_n_webcam = st.sidebar.slider("Webcam: tambah counter tiap N frame", 1, 30, 5, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Stabilizer")
stable_n = st.sidebar.slider("Vote N frames (stabilizer)", 1, 21, 9, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Webcam LIVE Performance")
skip_frames = st.sidebar.slider("Proses tiap N frame (lebih besar = lebih cepat)", 1, 10, 2, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Foto Performance")
max_side = st.sidebar.selectbox("Max sisi foto (px)", [0, 1280, 1024, 800], index=1)

st.sidebar.markdown("---")
local_save = st.sidebar.checkbox("Simpan output gambar/video ke folder lokal", value=False)
outdir_local = st.sidebar.text_input("Folder output gambar", value="outputs/pred_images")
outdir_video = st.sidebar.text_input("Folder output video", value="outputs/videos")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Semua (Checklist + Log)"):
    reset_all(classes)
    st.rerun()

with st.sidebar.expander("Lihat nama kelas model"):
    st.code(str(model.names), language="python")


# =========================
# MAIN LAYOUT
# =========================
left, right = st.columns([3, 1], gap="large")

with right:
    checklist_placeholder = st.empty()

with left:
    tabs = st.tabs(["🖼️ Foto", "🗂️ Banyak Foto / ZIP", "🎞️ Video", "📷 Webcam LIVE (WebRTC)", "🧾 Log & Export"])

    # =========================
    # TAB 1: SINGLE IMAGE
    # =========================
    with tabs[0]:
        st.subheader("🖼️ Deteksi Foto (Upload)")
        up = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

        if up is not None:
            img = Image.open(up).convert("RGB")
            img_rgb = np.array(img)
            img_rgb = shrink_rgb(img_rgb, int(max_side))

            c1, c2 = st.columns(2)
            with c1:
                st.image(Image.fromarray(img_rgb), caption="Input", use_container_width=True)

            if st.button("▶️ Deteksi Foto Ini", type="primary"):
                annotated_rgb, rows = infer_image_array(model, img_rgb, conf, iou, imgsz, max_det, keep_ids)

                with c2:
                    st.image(annotated_rgb, caption="Output", use_container_width=True)

                ulabels = unique_labels(rows)
                st.write("**Label (unik):**", ", ".join(ulabels) if ulabels else "-")

                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    log_add(source=f"image:{up.name}", rows=rows)
                    update_availability_image(ulabels)

                    if local_save:
                        ensure_dir(outdir_local)
                        out_path = Path(outdir_local) / f"pred_{Path(up.name).stem}.png"
                        Image.fromarray(annotated_rgb).save(out_path)
                        st.success(f"Tersimpan: {out_path}")

                    st.download_button(
                        "⬇️ Download hasil (PNG)",
                        data=image_to_bytes(annotated_rgb, "PNG"),
                        file_name=f"pred_{Path(up.name).stem}.png",
                        mime="image/png",
                    )
                else:
                    st.info("Tidak ada deteksi di atas threshold.")

    # =========================
    # TAB 2: MULTI IMAGES / ZIP
    # =========================
    with tabs[1]:
        st.subheader("🗂️ Banyak Foto (Multi Upload) atau ZIP")
        colA, colB = st.columns(2)
        with colA:
            ups = st.file_uploader(
                "Upload banyak gambar",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="multi_img",
            )
        with colB:
            zip_up = st.file_uploader("Upload ZIP gambar", type=["zip"], key="zip_img")

        images: List[Tuple[str, np.ndarray]] = []

        if zip_up is not None:
            zdata = zip_up.getvalue()
            with zipfile.ZipFile(io.BytesIO(zdata), "r") as z:
                for name in z.namelist():
                    if name.lower().endswith((".jpg", ".jpeg", ".png")):
                        b = z.read(name)
                        img = Image.open(io.BytesIO(b)).convert("RGB")
                        arr = np.array(img)
                        arr = shrink_rgb(arr, int(max_side))
                        images.append((Path(name).name, arr))

        if ups:
            for f in ups:
                img = Image.open(f).convert("RGB")
                arr = np.array(img)
                arr = shrink_rgb(arr, int(max_side))
                images.append((f.name, arr))

        if images:
            st.write(f"Total images: {len(images)}")
            if st.button("▶️ Proses semua", type="primary"):
                prog = st.progress(0)
                out_files: List[Tuple[str, bytes]] = []
                all_rows: List[Dict] = []

                if local_save:
                    ensure_dir(outdir_local)

                for i, (name, arr) in enumerate(images, start=1):
                    ann, rows = infer_image_array(model, arr, conf, iou, imgsz, max_det, keep_ids)
                    out_files.append((f"pred_{Path(name).stem}.png", image_to_bytes(ann)))

                    ulabels = unique_labels(rows)
                    if ulabels:
                        update_availability_image(ulabels)

                    if rows:
                        log_add(source=f"batch:{name}", rows=rows)
                        all_rows.extend([{"source": f"batch:{name}", **r} for r in rows])

                    if local_save:
                        Image.fromarray(ann).save(Path(outdir_local) / f"pred_{Path(name).stem}.png")

                    prog.progress(i / len(images))

                st.success("Selesai ✅")

                zbytes = zip_bytes_from_files(out_files)
                st.download_button(
                    "⬇️ Download semua hasil (ZIP)",
                    data=zbytes,
                    file_name="pred_images.zip",
                    mime="application/zip",
                )

                if all_rows:
                    df = pd.DataFrame(all_rows)
                    st.markdown("### Ringkasan label")
                    st.dataframe(
                        df.groupby("label")["conf"].agg(["count", "mean"]).sort_values("count", ascending=False),
                        use_container_width=True
                    )

    # =========================
    # TAB 3: VIDEO
    # =========================
    with tabs[2]:
        st.subheader("🎞️ Deteksi Video (Upload)")
        vup = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False)
        out_format = st.radio("Output format", ["AVI (XVID)", "MP4 (mp4v)"], horizontal=True)

        if vup is not None:
            st.video(vup)

            if st.button("▶️ Proses video", type="primary"):
                ensure_dir(outdir_video)
                tmpdir = Path(outdir_video) / "_tmp_upload"
                tmpdir.mkdir(parents=True, exist_ok=True)

                in_path = tmpdir / vup.name
                in_path.write_bytes(vup.getvalue())

                cap = cv2.VideoCapture(str(in_path))
                if not cap.isOpened():
                    st.error("Gagal membuka video input.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 1:
                    fps = 25.0

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

                stem = safe_stem(vup.name)
                if out_format.startswith("AVI"):
                    out_path = Path(outdir_video) / f"pred_{stem}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                else:
                    out_path = Path(outdir_video) / f"pred_{stem}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    cap.release()
                    st.error("VideoWriter gagal dibuka. Coba AVI (XVID).")
                    st.stop()

                prog = st.progress(0)
                status = st.empty()

                hist: List[Optional[str]] = []
                frame_idx = 0

                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame_idx += 1

                        results = model.predict(
                            source=frame,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            max_det=max_det,
                            classes=keep_ids,
                            verbose=False,
                        )
                        r = results[0]
                        annotated_bgr = r.plot(conf=True, labels=True)

                        rows: List[Dict] = []
                        if r.boxes is not None and len(r.boxes) > 0:
                            for b in r.boxes:
                                cid = int(b.cls.item())
                                score = float(b.conf.item())
                                rows.append({"label": model.names.get(cid, str(cid)), "conf": round(score, 4)})

                        bl = best_label(rows)
                        lbl = bl[0] if bl else None

                        hist.append(lbl)
                        if len(hist) > stable_n:
                            hist = hist[-stable_n:]
                        series = [x for x in hist if x is not None]
                        stable = max(set(series), key=series.count) if series else None

                        update_availability_video(stable, frame_idx, every_n_video)

                        if stable:
                            cv2.putText(
                                annotated_bgr, f"STABLE: {stable}",
                                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 0), 2, cv2.LINE_AA
                            )

                        writer.write(annotated_bgr)

                        if bl:
                            st.session_state["log_rows"].append({
                                "source": f"video:{vup.name}@{frame_idx}",
                                "label": bl[0],
                                "conf": round(bl[1], 4),
                                "x1": None, "y1": None, "x2": None, "y2": None,
                            })

                        if total > 0:
                            prog.progress(min(frame_idx / total, 1.0))
                            status.write(f"Frame {frame_idx}/{total}")
                        else:
                            status.write(f"Frame {frame_idx}")

                finally:
                    cap.release()
                    writer.release()

                st.success(f"Selesai ✅ tersimpan: {out_path}")

                st.download_button(
                    "⬇️ Download video hasil",
                    data=Path(out_path).read_bytes(),
                    file_name=out_path.name,
                    mime="video/mp4" if out_path.suffix.lower() == ".mp4" else "video/x-msvideo",
                )

    # =========================
    # TAB 4: WEBCAM LIVE (WEBRTC)
    # =========================
    with tabs[3]:
        st.subheader("📷 Webcam LIVE (WebRTC)")
        st.caption("Klik START → izinkan kamera → video live akan diproses frame-by-frame.")

        rtc_conf = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }

        def processor_factory():
            return YOLOWebRTCProcessor(
                model=model,
                model_names=model.names,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                keep_ids=keep_ids,
                stable_n=stable_n,
                every_n_webcam=every_n_webcam,
                skip_frames=skip_frames,
            )

        webrtc_ctx = webrtc_streamer(
            key="yolo-live",
            video_processor_factory=processor_factory,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=rtc_conf,
            async_processing=True,
        )

        col1, col2, col3 = st.columns([2, 2, 2])

        # Tombol manual untuk "tarik" status terbaru + update checklist/log
        if st.button("🔄 Sync status webcam → Checklist/Log"):
            vp = webrtc_ctx.video_processor if webrtc_ctx else None
            if vp is None:
                st.warning("Webcam belum START atau belum siap.")
            else:
                with vp.lock:
                    last_best = vp.last_best
                    last_stable = vp.last_stable
                    frame_idx = vp.frame_idx

                if last_best:
                    st.session_state["log_rows"].append({
                        "source": f"webcam@{frame_idx}",
                        "label": last_best[0],
                        "conf": round(float(last_best[1]), 4),
                        "x1": None, "y1": None, "x2": None, "y2": None,
                    })

                # update checklist (pakai label stable biar tidak noise)
                update_availability_webcam(last_stable, frame_idx, every_n_webcam)

                st.success(f"Synced ✅ stable={last_stable if last_stable else '-'} | best={last_best[0] if last_best else '-'}")

        vp2 = webrtc_ctx.video_processor if webrtc_ctx else None
        if vp2 is not None:
            with vp2.lock:
                lb = vp2.last_best
                stbl = vp2.last_stable
                fidx = vp2.frame_idx

            with col1:
                st.metric("Frame processed", fidx)
            with col2:
                st.metric("Stable label", stbl if stbl else "-")
            with col3:
                st.metric("Best conf", f"{lb[1]:.2f}" if lb else "-")

        st.info("Kalau terasa berat/lag: naikkan 'Proses tiap N frame' atau turunkan imgsz.")

    # =========================
    # TAB 5: LOG & EXPORT
    # =========================
    with tabs[4]:
        st.subheader("🧾 Log Deteksi & Export")
        rows = st.session_state["log_rows"]
        kpi_cards(rows)

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=420)

            st.markdown("### Ringkasan label (log)")
            st.dataframe(
                df.groupby("label")["conf"].agg(["count", "mean"]).sort_values("count", ascending=False),
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Log CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="detections_log.csv",
                mime="text/csv",
            )

            if st.button("🧹 Clear log"):
                st.session_state["log_rows"] = []
                st.success("Log cleared.")
        else:
            st.info("Log masih kosong. Jalankan deteksi dulu dari Foto/Video/Webcam.")


# =========================
# RENDER CHECKLIST DI AKHIR
# =========================
with checklist_placeholder.container():
    render_availability_panel(classes, stream_threshold)
