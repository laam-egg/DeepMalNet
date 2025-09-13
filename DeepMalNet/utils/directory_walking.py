import os

def walk_files_in_dir(root_dir):
    for root, dirs, files in os.walk(root_dir, followlinks=True):
        for file in files:
            yield os.path.abspath(os.path.join(root, file))
