import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\absensi\dataset_raw", help="Root dataset yang berisi train/")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Proporsi validasi (0.2 = 20%)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["move", "copy"], default="move",
                    help="move = pindahkan file ke valid (train jadi bersih), copy = duplikasi (kurang disarankan)")
    args = ap.parse_args()

    root = Path(args.root)
    train_img = root / "train" / "images"
    train_lbl = root / "train" / "labels"
    val_img = root / "valid" / "images"
    val_lbl = root / "valid" / "labels"

    if not train_img.exists():
        raise FileNotFoundError(f"Tidak ada: {train_img}")
    if not train_lbl.exists():
        raise FileNotFoundError(f"Tidak ada: {train_lbl}")

    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    images = [p for p in train_img.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise RuntimeError(f"Tidak ada gambar di {train_img}")

    random.seed(args.seed)
    random.shuffle(images)

    n_val = max(1, int(round(len(images) * args.val_ratio)))
    pick = images[:n_val]

    moved = 0
    empty_labels = 0

    for img_path in pick:
        stem = img_path.stem
        lbl_path = train_lbl / f"{stem}.txt"

        dst_img = val_img / img_path.name
        dst_lbl = val_lbl / f"{stem}.txt"

        if args.mode == "move":
            shutil.move(str(img_path), str(dst_img))
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(dst_lbl))
            else:
                dst_lbl.write_text("", encoding="utf-8")
                empty_labels += 1
        else:  # copy
            shutil.copy2(str(img_path), str(dst_img))
            if lbl_path.exists():
                shutil.copy2(str(lbl_path), str(dst_lbl))
            else:
                dst_lbl.write_text("", encoding="utf-8")
                empty_labels += 1

        moved += 1

    print("OK ✅ Valid split dibuat")
    print("Root :", root)
    print("Train images tersisa:", len([p for p in train_img.iterdir() if p.suffix.lower() in IMG_EXTS]))
    print("Valid images:", len(list(val_img.iterdir())))
    print("Empty label dibuat:", empty_labels)
    print("Mode:", args.mode)

if __name__ == "__main__":
    main()
