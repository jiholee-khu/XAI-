# extract_features.py

import os
import subprocess
import numpy as np
from tqdm import tqdm
from config import Config

OPENSENSE_BIN = "/usr/local/bin/SMILExtract"  # 너희 서버 경로로 수정
CONFIG_FILE = "./opensmile/config/gemaps/eGeMAPSv02.conf"

def extract_egemaps(wav_path, out_csv):
    cmd = [
        OPENSENSE_BIN,
        "-C", CONFIG_FILE,
        "-I", wav_path,
        "-O", out_csv
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def csv_to_npy(csv_path):
    with open(csv_path, "r") as f:
        lines = f.readlines()
    last = lines[-1].strip().split(",")[1:]
    return np.array(last, dtype=np.float32)

if __name__ == "__main__":
    os.makedirs(Config.audio_feature_dir, exist_ok=True)

    for root, dirs, files in os.walk(Config.ravdess_root):
        for f in files:
            if not f.endswith(".wav"):
                continue

            wav_path = os.path.join(root, f)
            audio_id = f.replace(".wav", "")

            out_csv = f"{Config.audio_feature_dir}/{audio_id}.csv"
            out_npy = f"{Config.audio_feature_dir}/{audio_id}.npy"

            extract_egemaps(wav_path, out_csv)
            arr = csv_to_npy(out_csv)
            np.save(out_npy, arr)
            os.remove(out_csv)

            print("[OK]", audio_id)
