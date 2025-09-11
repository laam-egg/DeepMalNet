import baker
from .training import Trainer
import os
import sys

@baker.command
def train(lmdb_path):
    # type: (str) -> None

    CHECKPOINTS_DIR = os.path.join(os.path.dirname(sys.argv[0]), "../checkpoints")
    trainer = Trainer(lmdb_path)
    trainer.load_last_checkpoint(CHECKPOINTS_DIR)
    trainer.train()
    trainer.save(CHECKPOINTS_DIR)

@baker.command
def infer():
    raise NotImplementedError()

if __name__ == "__main__":
    baker.run()
