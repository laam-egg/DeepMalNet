import baker
from .training import Trainer

@baker.command
def train():
    trainer = Trainer()
    trainer.train()
    trainer.save()

@baker.command
def infer():
    raise NotImplementedError()

if __name__ == "__main__":
    baker.run()
