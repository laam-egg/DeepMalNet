from ..datasets.PEFEDataset import PEFEDataset
from ..models.DeepMalNetModel import DeepMalNetNNModule, DeepMalNet_Mode
from ..models.DeepMalNetModel.ember2024 import NUM_FEATURES

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, lmdb_path):
        # type: (Trainer, str) -> None
        print(f"Initializing model training...")
        self.NUM_EPOCHS = 128
        self.lmdb_path = lmdb_path
        self.model = DeepMalNetNNModule(NUM_FEATURES, DeepMalNet_Mode.TRAINING)
        self.transfer_model_to_accelerator()
        self.initialize_loss_and_optimizer()

    def train(self):
        train_loader = DataLoader(
            PEFEDataset(self.lmdb_path, "train"),
            batch_size=256,
            shuffle=True,
            num_workers=4,
        )

        cv_loader = DataLoader(
            PEFEDataset(self.lmdb_path, "cv"),
            batch_size=256,
            shuffle=False,
            num_workers=4,
        )

        for epoch in range(self.NUM_EPOCHS):
            self.last_epoch = epoch
            self.model.train()
            total_loss = 0
            
            for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.NUM_EPOCHS} :: T"):
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
                for X, y in tqdm(cv_loader, desc=f"Epoch {epoch+1}/{self.NUM_EPOCHS} :: V"):
                    X, y = (
                        X.to(self.DEVICE_NAME),
                        y.to(self.DEVICE_NAME).unsqueeze(1),
                    )
                    logits = self.model(X)
                    preds = torch.sigmoid(logits) > 0.5
                    correct += (preds == y).sum().item()
                    total   += y.size(0)
            acc = correct / total

            print(f"Epoch {epoch+1}/{self.NUM_EPOCHS}: train loss={avg_loss:.4f}, val acc={acc:.4f}")
    
    def save(self):
        import sys, os, time
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.last_epoch,
                "loss": self.last_loss,
            },

            os.path.dirname(sys.argv[0])
            + "/checkpoints/" + str(time.time()) + ".pth"
        )

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
