import hashlib

def hash_file(filepath, algorithm="sha256", chunk_size=8192):
    # Get a new hash object (e.g. sha256, md5, sha1, etc.)
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
