# config.py

class Config:
    # Text model
    model_name = "bert-base-uncased"
    max_len = 64

    # Training settings
    batch_size = 16
    lr_text = 3e-5
    lr_audio = 1e-3
    lr_fusion = 1e-4

    num_epochs_text = 5
    num_epochs_audio = 20
    num_epochs_fusion = 10

    # Paths
    data_csv = "./data/metadata.csv"
    audio_feature_dir = "./data/features"
    ravdess_root = "./data/ravdess"
    save_dir = "./outputs/checkpoints"
