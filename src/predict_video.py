import argparse
from pathlib import Path
from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="weights/best_items.pt")
    ap.add_argument("--source", type=str, required=True, help="Path video, contoh: D:\\absensi\\video\\test.mp4")
    ap.add_argument("--outdir", type=str, required=True, help="Folder output, contoh: D:\\absensi\\hasil\\prediksi")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--show", action="store_true", help="Tampilkan video realtime")
    args = ap.parse_args()

    weights = Path(args.weights)
    source = Path(args.source)
    outdir = Path(args.outdir)

    if not weights.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {source}")
    if source.suffix.lower() not in VIDEO_EXTS:
        print(f"Warning: ekstensi {source.suffix} bukan ekstensi video umum, tapi tetap dicoba...")

    outdir.mkdir(parents=True, exist_ok=True)

    # Ultralytics simpan output via project/name
    project = str(outdir.parent)  # mis: D:\absensi\hasil
    name = outdir.name            # mis: prediksi

    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,           # simpan output
        show=args.show,      # tampilkan kalau mau
        project=project,
        name=name,
        exist_ok=True
    )

    print(f"Selesai. Output video ada di: {outdir}")

if __name__ == "__main__":
    main()
