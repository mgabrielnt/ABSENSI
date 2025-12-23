import argparse
from pathlib import Path
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="weights/best_items.pt")
    ap.add_argument("--source_dir", type=str, required=True, help="Folder input gambar")
    ap.add_argument("--outdir", type=str, required=True, help="Folder output (mis. D:\\absensi\\hasil\\prediksi)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    weights = Path(args.weights)
    source_dir = Path(args.source_dir)
    outdir = Path(args.outdir)

    if not weights.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {weights}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Folder input tidak ditemukan: {source_dir}")

    outdir.mkdir(parents=True, exist_ok=True)

    images = []
    for p in source_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(str(p))

    print(f"Ketemu {len(images)} gambar dari: {source_dir}")
    if len(images) == 0:
        raise RuntimeError(f"Tidak ada gambar di: {source_dir}")

    # Simpan tepat ke outdir dengan project/name
    project = str(outdir.parent)   # D:\absensi\hasil
    name = outdir.name            # prediksi

    model = YOLO(str(weights))
    model.predict(
        source=images,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        show=args.show,
        project=project,
        name=name,
        exist_ok=True
    )

    print(f"Selesai. Output tersimpan di: {outdir}")

if __name__ == "__main__":
    main()
