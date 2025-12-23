import os
from roboflow import Roboflow

print("Starting download...")

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
print("API key set?", bool(API_KEY))

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("viskom-szwla").project("my-first-project-sbky7")
version = project.version(1)

print("Downloading to D:\\absensi\\dataset_raw ...")
dataset = version.download("yolov8", location=r"D:\absensi\dataset_raw")

print("DONE")
print("Dataset location:", dataset.location)
