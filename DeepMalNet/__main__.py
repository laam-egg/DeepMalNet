import baker
from .training import Trainer

@baker.command
def train(lmdb_path):
    # type: (str) -> None
    trainer = Trainer(lmdb_path)
    trainer.train()
    trainer.save()

@baker.command
def infer():
    raise NotImplementedError()

if __name__ == "__main__":
    baker.run()
