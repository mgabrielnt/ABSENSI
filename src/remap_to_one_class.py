import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True, help=r"contoh: D:\absensi\dataset\labels")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)

    txts = list(labels_dir.rglob("*.txt"))
    print("Total label files:", len(txts))

    for p in txts:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            # YOLO format: class x y w h
            parts[0] = "0"
            new_lines.append(" ".join(parts))
        p.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    print("Done remap all to class 0.")

if __name__ == "__main__":
    main()
