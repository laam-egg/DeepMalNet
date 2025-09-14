from ..datasets.PEFEDataset import PEFEDataset
from ..models.DeepMalNetModel import DeepMalNetNNModule, DeepMalNet_Mode
from ..models.DeepMalNetModel.ember2024 import NUM_FEATURES

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import re
import time
from ..utils import hash_file, model_sanity_check, select_device
from ..utils.loss import FocalLoss

class Trainer:
    def __init__(self, lmdb_path, num_epochs=4):
        # type: (Trainer, str) -> None
        print(f"Initializing model training...")
        self.NUM_EPOCHS_TO_RUN_THIS_TIME = num_epochs
        self.lmdb_path = lmdb_path
        self.model = DeepMalNetNNModule(NUM_FEATURES, DeepMalNet_Mode.TRAINING)
        self.DEVICE_NAME = select_device()
        self._transfer_model_to_accelerator()
        self._initialize_loss_and_optimizer()
        self._create_data_loaders()
        self.last_epoch = 0
        self.last_loss = float('nan')
    
    CHECKPOINT_FILENAME_PATTERN = re.compile(r"^epoch(\d+)_(\d+\.\d+)\.pth$") # e.g. epoch32_123456.789.pth
    def load_last_checkpoint(self, checkpoint_dir, sanity_check_if_found=False):
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
        self.model.load_model_data(checkpoint)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.last_epoch = checkpoint["epoch"]
        self.last_loss = checkpoint.get("loss", float("nan"))

        print(f"[INFO] Loaded checkpoint '{os.path.basename(path)}' "
              f"(epoch={epoch}, time={t}, loss={self.last_loss:.4f})")
        
        if sanity_check_if_found:
            self.sanity_check()
    
    def sanity_check(self):
        print(f"Checking model sanity...")
        model_sanity_check(self.model, self.cv_loader, self.DEVICE_NAME)
        print(f"Checking model sanity: Done.")

    def train(self):
        train_loader = self.train_loader
        cv_loader = self.cv_loader

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

                # Preventing runaway updates with gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)

            self.last_loss = avg_loss

            self.scheduler.step()

            # Validation

            self.model.eval()
            correct, total = 0, 0
            TP, FP, FN, TN = 0, 0, 0, 0

            with torch.no_grad():
                for X, y in tqdm(cv_loader, desc=f"Epoch {epoch} (run {i_run+1}/{self.NUM_EPOCHS_TO_RUN_THIS_TIME}) :: V"):
                    X, y = (
                        X.to(self.DEVICE_NAME),
                        y.to(self.DEVICE_NAME),
                    )
                    logits = self.model(X)
                    preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
                    y = y.long()

                    correct += (preds == y).sum().item()
                    total += y.size(0)
                    TP += ((preds == 1) & (y == 1)).sum().item()
                    FP += ((preds == 1) & (y == 0)).sum().item()
                    FN += ((preds == 0) & (y == 1)).sum().item()
                    TN += ((preds == 0) & (y == 0)).sum().item()
            
            acc = correct / total if total > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            _pPLUSr = (precision + recall)
            f1 = (2 * precision * recall) / _pPLUSr if _pPLUSr > 0 else 0.0

            print(f"Epoch {epoch} (run {i_run+1}/{self.NUM_EPOCHS_TO_RUN_THIS_TIME}) : train loss={avg_loss:.4f} | val acc={acc:.4f}, f1={f1:.4f}, "
                + f"precision={precision:.4f}, recall={recall:.4f}, TP={TP} FP={FP} FN={FN} TN={TN}, "
                + f"correct={correct}, total={total}"
            )
    
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
                "scheduler_state_dict": self.scheduler.state_dict(),
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

    def _initialize_loss_and_optimizer(self):
        # num_neg = 56.81
        # num_pos = 43.19
        # weight = torch.tensor([num_neg / num_pos], device=self.DEVICE_NAME)
        # assumes model output is raw score (not sigmoid)
        # self.criterion = nn.BCEWithLogitsLoss(
        #     pos_weight=weight,
        # )
        self.criterion = FocalLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=2e-4,
            weight_decay=1e-2,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.NUM_EPOCHS_TO_RUN_THIS_TIME,
            eta_min=1e-6, # minimum learning rate
        )
    
    def _transfer_model_to_accelerator(self):
        DEVICE_NAME = self.DEVICE_NAME
        try:
            self.model.to(DEVICE_NAME)
        except Exception as e:
            print(f"Failed to use device {DEVICE_NAME}: {e}")
            print(f"Falling back to CPU-only mode.")
            DEVICE_NAME = "cpu"

        self.DEVICE_NAME = DEVICE_NAME
        print(f"Operating on device: {self.DEVICE_NAME}")
    
    def _create_data_loaders(self):
        NUM_WORKERS = max(1, min(4, os.cpu_count() or 1))

        self.train_loader = DataLoader(
            PEFEDataset(self.lmdb_path, "train"),
            batch_size=256,
            shuffle=True,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )

        self.cv_loader = DataLoader(
            PEFEDataset(self.lmdb_path, "cv"),
            batch_size=256,
            shuffle=False,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )
