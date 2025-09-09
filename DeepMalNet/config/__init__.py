from pefe_common.config import Config

config = Config.load({
    "ember2024_lmdb_path": (str, None),
}, "config.json")
