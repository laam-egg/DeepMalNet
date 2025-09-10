import thrember
import sys, os

SCRIPT_DIR = os.path.dirname(sys.argv[0])
DATASET_DIR = SCRIPT_DIR + "/../dataset/EMBER2024/"

os.makedirs(DATASET_DIR, exist_ok=True)

thrember.download_dataset(DATASET_DIR, file_type="PE")
