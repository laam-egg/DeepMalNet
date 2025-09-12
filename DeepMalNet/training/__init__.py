from ..datasets.PEFEDataset import PEFEDataset
from ..models.DeepMalNetModel import DeepMalNetNNModule, DeepMalNet_Mode
from ..models.DeepMalNetModel.ember2024 import NUM_FEATURES

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader

import os
import re
import time
from ..utils import hash_file

class Trainer:
    def __init__(self, lmdb_path, num_epochs=4):
        # type: (Trainer, str) -> None
        print(f"Initializing model training...")
        self.NUM_EPOCHS_TO_RUN_THIS_TIME = num_epochs
        self.lmdb_path = lmdb_path
        self.model = DeepMalNetNNModule(NUM_FEATURES, DeepMalNet_Mode.TRAINING)
        self.transfer_model_to_accelerator()
        self.initialize_loss_and_optimizer()
        self.last_epoch = 0
        self.last_loss = float('nan')
    
    CHECKPOINT_FILENAME_PATTERN = re.compile(r"epoch(\d+)_(\d+\.\d+)\.pth") # e.g. epoch32_123456.789.pth
    def load_last_checkpoint(self, checkpoint_dir):
        """
        Load the checkpoint with the highest epoch. 
        If multiple have the same epoch, pick the one with the latest TIME value.
        """
        pattern = self.CHECKPOINT_FILENAME_PATTERN

        if not os.path.isdir(checkpoint_dir):
            print(f"[WARN] Checkpoint directory not found: {checkpoint_dir}")
            self.last_epoch = 0
            return

        all_files = os.listdir(checkpoint_dir)
        print(f"[INFO] Found {len(all_files)} files in {checkpoint_dir}:")
        for f in all_files:
            print("   ", f)

        candidates = []
        for fname in all_files:
            m = pattern.match(fname)
            if m:
                epoch = int(m.group(1))
                t = float(m.group(2))
                candidates.append((epoch, t, os.path.join(checkpoint_dir, fname)))

        if not candidates:
            print(f"[WARN] No checkpoint files matched pattern in {checkpoint_dir}")
            self.last_epoch = 0
            return

        # Sort by epoch first, then TIME
        best = max(candidates, key=lambda x: (x[0], x[1]))
        epoch, t, path = best

        print(f"[INFO] Loading checkpoint: {path}")

        checkpoint = torch.load(path, map_location=self.DEVICE_NAME)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.last_epoch = checkpoint["epoch"]
        self.last_loss = checkpoint.get("loss", float("nan"))

        print(f"[INFO] Loaded checkpoint '{os.path.basename(path)}' "
              f"(epoch={epoch}, time={t}, loss={self.last_loss:.4f})")

    def train(self):
        NUM_WORKERS = max(1, min(4, os.cpu_count() or 1))

        train_loader = DataLoader(
            PEFEDataset(self.lmdb_path, "train"),
            batch_size=256,
            shuffle=True,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )

        cv_loader = DataLoader(
            PEFEDataset(self.lmdb_path, "cv"),
            batch_size=256,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )

        for i_run in range(self.NUM_EPOCHS_TO_RUN_THIS_TIME):
            self.last_epoch += 1
            epoch = self.last_epoch
            self.model.train()
            total_loss = 0
            
            for X, y in tqdm(train_loader, desc=f"Epoch {epoch} (run {i_run+1}/{self.NUM_EPOCHS_TO_RUN_THIS_TIME}) :: T"):
                X, y = (
                    X.to(self.DEVICE_NAME),
                    y.to(self.DEVICE_NAME).unsqueeze(1),  # shape (batch,1)
                )

                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)

            self.last_loss = avg_loss

            # Validation

            self.model.eval()
            correct, total = 0, 0

            with torch.no_grad():
                for X, y in tqdm(cv_loader, desc=f"Epoch {epoch} (run {i_run+1}/{self.NUM_EPOCHS_TO_RUN_THIS_TIME}) :: V"):
                    X, y = (
                        X.to(self.DEVICE_NAME),
                        y.to(self.DEVICE_NAME).unsqueeze(1),
                    )
                    logits = self.model(X)
                    preds = (torch.sigmoid(logits) > 0.5).long()
                    correct += (preds.squeeze(1) == y.long()).sum().item()
                    total += y.size(0)
            acc = correct / total if total > 0 else 0.0

            print(f"Epoch {epoch} (run {i_run+1}/{self.NUM_EPOCHS_TO_RUN_THIS_TIME}) : train loss={avg_loss:.4f}, val acc={acc:.4f}")
    
    def save(self, checkpoints_dir):
        # type: (Trainer, str) -> None

        os.makedirs(checkpoints_dir, exist_ok=True)

        CHECKPOINT_PATH = os.path.join(
            checkpoints_dir,
            f"epoch{self.last_epoch}_{time.time()}.pth",
        )

        CHECKPOINT_HASH_PATH = CHECKPOINT_PATH + ".sha256.txt"

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.last_epoch,
                "loss": self.last_loss,
            },

            CHECKPOINT_PATH,
        )

        print(f"[INFO] Saved checkpoint: '{CHECKPOINT_PATH}'")

        h = hash_file(CHECKPOINT_PATH, algorithm="sha256")
        print(f"[INFO] Checkpoint SHA256: {h}")
        with open(CHECKPOINT_HASH_PATH, 'w') as hf:
            hf.write(h)

    def initialize_loss_and_optimizer(self):
        self.criterion = nn.BCEWithLogitsLoss()   # assumes model output is raw score (not sigmoid)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def transfer_model_to_accelerator(self):
        DEVICE_NAME = "cuda:0"
        try:
            self.model.to(DEVICE_NAME)
        except Exception as e:
            print(f"Failed to use device {DEVICE_NAME}: {e}")
            print(f"Falling back to CPU-only mode.")
            DEVICE_NAME = "cpu"

        self.DEVICE_NAME = DEVICE_NAME
        print(f"Operating on device: {self.DEVICE_NAME}")
