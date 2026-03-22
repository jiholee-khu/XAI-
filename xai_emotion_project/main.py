# main.py

import os
import json
import torch
import shap
import numpy as np
import pandas as pd
from tqdm import tqdm
from lime.lime_tabular import LimeTabularExplainer

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL2ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ================================
# Dataset
# ================================
class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, feature_dir):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        text = r["text"]
        audio_id = r["audio_id"]

        enc = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )

        feat = np.load(f"{self.feature_dir}/{audio_id}.npy")

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "audio": torch.tensor(feat, dtype=torch.float32),
            "label": torch.tensor(LABEL2ID[r["label"]]),
            "text_raw": text,
        }

# ================================
# Models
# ================================
class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(Config.model_name)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(self.bert.config.hidden_size, 4)

    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls), cls


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(88, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.text = text_model
        self.text.classifier = nn.Identity()  # Remove text classifier
        self.mlp = nn.Sequential(
            nn.Linear(768 + 88, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )

    def forward(self, ids, mask, audio):
        with torch.no_grad():
            out = self.text.bert(ids, attention_mask=mask)
            cls = out.last_hidden_state[:, 0]
        f = torch.cat([cls, audio], dim=1)
        return self.mlp(f)

# ================================
# Train / Eval
# ================================
def train_epoch(model, loader, opt, criterion, mode):
    model.train()
    total, correct = 0, 0

    for batch in tqdm(loader):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        audio = batch["audio"].to(DEVICE)
        lbl = batch["label"].to(DEVICE)

        opt.zero_grad()

        if mode == "text":
            out, _ = model(ids, mask)
        elif mode == "audio":
            out = model(audio)
        else:
            out = model(ids, mask, audio)

        loss = criterion(out, lbl)
        loss.backward()
        opt.step()

        pred = out.argmax(1)
        correct += (pred == lbl).sum().item()
        total += lbl.size(0)

    return correct / total

def eval_epoch(model, loader, criterion, mode):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch in tqdm(loader):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            lbl = batch["label"].to(DEVICE)

            if mode == "text":
                out, _ = model(ids, mask)
            elif mode == "audio":
                out = model(audio)
            else:
                out = model(ids, mask, audio)

            pred = out.argmax(1)
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)

    return correct / total

# ================================
# Masking functions
# ================================
PITCH = list(range(0,20))
ENERGY = list(range(20,31))
JITTER = [31,32]
SHIMMER = [33,34]

def mask_audio(audio, group):
    a = audio.clone()
    if group == "pitch": a[:, PITCH] = 0
    if group == "energy": a[:, ENERGY] = 0
    if group == "jitter": a[:, JITTER] = 0
    if group == "shimmer": a[:, SHIMMER] = 0
    return a

# ================================
# Main
# ================================
def main():
    os.makedirs(Config.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    df = pd.read_csv(Config.data_csv)
    train_df = df[df.split=="train"]
    valid_df = df[df.split=="valid"]
    test_df  = df[df.split=="test"]

    train_loader = DataLoader(EmotionDataset(train_df, tokenizer, Config.max_len, Config.audio_feature_dir), batch_size=Config.batch_size, shuffle=True)
    valid_loader = DataLoader(EmotionDataset(valid_df, tokenizer, Config.max_len, Config.audio_feature_dir), batch_size=Config.batch_size)
    test_loader  = DataLoader(EmotionDataset(test_df,  tokenizer, Config.max_len, Config.audio_feature_dir), batch_size=Config.batch_size)

    # ===== Text Model =====
    text_model = TextModel().to(DEVICE)
    opt_text = torch.optim.Adam(text_model.classifier.parameters(), lr=Config.lr_text)
    criterion = nn.CrossEntropyLoss()

    for e in range(Config.num_epochs_text):
        acc = train_epoch(text_model, train_loader, opt_text, criterion, "text")
        val = eval_epoch(text_model, valid_loader, criterion, "text")
        print(f"[TEXT] epoch {e+1} acc={acc:.3f} val={val:.3f}")

    # ===== Audio Model =====
    audio_model = AudioModel().to(DEVICE)
    opt_audio = torch.optim.Adam(audio_model.parameters(), lr=Config.lr_audio)

    for e in range(Config.num_epochs_audio):
        acc = train_epoch(audio_model, train_loader, opt_audio, criterion, "audio")
        val = eval_epoch(audio_model, valid_loader, criterion, "audio")
        print(f"[AUDIO] epoch {e+1} acc={acc:.3f} val={val:.3f}")

    # ===== Fusion Model =====
    fusion_model = FusionModel(text_model).to(DEVICE)
    opt_fusion = torch.optim.Adam(fusion_model.mlp.parameters(), lr=Config.lr_fusion)

    for e in range(Config.num_epochs_fusion):
        acc = train_epoch(fusion_model, train_loader, opt_fusion, criterion, "fusion")
        val = eval_epoch(fusion_model, valid_loader, criterion, "fusion")
        print(f"[FUSION] epoch {e+1} acc={acc:.3f} val={val:.3f}")

    # Save
    torch.save(fusion_model.state_dict(), f"{Config.save_dir}/fusion.pt")

    # ===== Masking Experiment =====
    print("---- MASKING ----")
    base = eval_epoch(fusion_model, test_loader, criterion, "fusion")
    print("Baseline:", base)

    for g in ["pitch", "energy", "jitter", "shimmer"]:
        tot, cor = 0, 0
        fusion_model.eval()
        with torch.no_grad():
            for batch in test_loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                audio = batch["audio"].to(DEVICE)
                lbl = batch["label"].to(DEVICE)

                masked = mask_audio(audio, g)
                out = fusion_model(ids, mask, masked)
                pred = out.argmax(1)
                cor += (pred == lbl).sum().item()
                tot += lbl.size(0)
        print(f"{g}: {cor/tot:.3f}")

if __name__ == "__main__":
    main()
