# make_metadata.py

import os
import pandas as pd

RAVDESS_ROOT = "./data/ravdess"
OUTPUT_CSV = "./data/metadata.csv"

emotion_map = {
    "01": "neutral",
    "02": "neutral",   # calm → neutral 병합
    "03": "happy",
    "04": "sad",
    "05": "angry",
}

def parse_emotion(file_name):
    parts = file_name.split("-")
    emotion_id = parts[2]  # third field
    return emotion_map.get(emotion_id, None)

def parse_text(file_name):
    parts = file_name.split("-")
    stmt_id = parts[4]   # statement number: 01 or 02
    if stmt_id == "01":
        return "Kids are talking by the door."
    else:
        return "Dogs are sitting by the door."

def main():
    rows = []

    for actor in os.listdir(RAVDESS_ROOT):
        actor_dir = os.path.join(RAVDESS_ROOT, actor)
        if not os.path.isdir(actor_dir):
            continue

        for f in os.listdir(actor_dir):
            if not f.endswith(".wav"):
                continue

            audio_id = f.replace(".wav", "")

            label = parse_emotion(audio_id)
            if label is None:
                continue  # skip fearful/disgust/surprised

            text = parse_text(audio_id)

            # default ambiguous = 1 (같은 문장이 여러 감정)
            is_ambiguous = 1

            # split rule (8:1:1 by actor ID)
            actor_num = int(audio_id.split("-")[-1])
            if actor_num <= 19:
                split = "train"
            elif actor_num == 20 or actor_num == 21:
                split = "valid"
            else:
                split = "test"

            rows.append([
                audio_id,
                text,
                label,
                split,
                is_ambiguous
            ])

    df = pd.DataFrame(rows, columns=[
        "audio_id", "text", "label", "split", "is_ambiguous"
    ])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"metadata.csv 생성 완료! → {OUTPUT_CSV}")
    print(f"총 샘플 수: {len(df)}")


if __name__ == "__main__":
    main()
