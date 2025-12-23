import argparse
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="weights/best_items.pt") # Pastikan path ini benar
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--outdir", type=str, default="hasil_absensi")
    args = ap.parse_args()

    # Setup
    model = YOLO(args.weights)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # === LOGIKA PRESENSI ===
    # Set untuk menyimpan siapa yang sudah absen hari ini agar tidak duplikat
    sudah_absen = set() 
    log_data = []
    filename_csv = outdir / f"presensi_{datetime.now().strftime('%Y-%m-%d')}.csv"

    print(f"Mulai Sistem Presensi... Tekan 'q' di jendela kamera untuk berhenti.")
    
    # Buka Webcam pakai OpenCV (Lebih stabil drpd mode stream CLI)
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280) # Lebar
    cap.set(4, 720)  # Tinggi

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Inferensi
        results = model(frame, stream=True, conf=args.conf, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Ambil Nama
                cls_id = int(box.cls[0])
                nama = model.names[cls_id]
                conf = float(box.conf[0])

                # Gambar Kotak Visual
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{nama} {conf:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # === REKAM DATA ===
                # Jika nama valid (bukan background) dan belum absen
                if nama not in sudah_absen and nama != "background":
                    waktu = datetime.now().strftime("%H:%M:%S")
                    sudah_absen.add(nama)
                    log_data.append({"Nama": nama, "Waktu": waktu, "Status": "Hadir"})
                    
                    print(f"✅ [PRESENSI MASUK] {nama} pada {waktu}")
                    
                    # Simpan Real-time ke CSV (agar kalau crash data aman)
                    df = pd.DataFrame(log_data)
                    df.to_csv(filename_csv, index=False)

        # Tampilkan Status di Layar
        cv2.putText(frame, f"Total Hadir: {len(sudah_absen)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Sistem Presensi Real-time", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Selesai. Data tersimpan di: {filename_csv}")

if __name__ == "__main__":
    main()