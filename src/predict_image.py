import argparse
import os
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="weights/best_items.pt")
    ap.add_argument("--source", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--project", type=str, default="runs/detect")  # root output
    ap.add_argument("--name", type=str, default="predict")         # subfolder output
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model tidak ditemukan: {args.weights}")
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"File tidak ditemukan: {args.source}")

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        show=True,
        save=args.save,
        project=args.project,
        name=args.name
    )

if __name__ == "__main__":
    main()
