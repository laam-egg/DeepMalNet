import baker
import os
import sys
from .utils import walk_files_in_dir

@baker.command
def train(lmdb_path):
    # type: (str) -> None
    from .training import Trainer

    CHECKPOINTS_DIR = os.path.join(os.path.dirname(sys.argv[0]), "../checkpoints")
    trainer = Trainer(lmdb_path)
    trainer.load_last_checkpoint(
        CHECKPOINTS_DIR,
        sanity_check_if_found=True, # optional
    )
    trainer.train()
    trainer.save(CHECKPOINTS_DIR)
    trainer.sanity_check() # optional

@baker.command
def infer(checkpoint_path, dir_or_file_path):
    from .models.DeepMalNetModel import InferenceDeepMalNetModel
    model = InferenceDeepMalNetModel()
    model.load(checkpoint_path)

    def predict(file_path):
        nonlocal model
        prob = model.predict_single_file(file_path)
        label = (
            "MALICIOUS" if prob >= 0.5 else
            "BENIGN   "
        )
        print(label, round(prob, 2), file_path)

    if os.path.isdir(dir_or_file_path):
        print(f"Walking in directory: {dir_or_file_path}")
        for file_path in walk_files_in_dir(dir_or_file_path):
            predict(file_path)
    else:
        file_path = dir_or_file_path
        predict(file_path)

if __name__ == "__main__":
    baker.run()
