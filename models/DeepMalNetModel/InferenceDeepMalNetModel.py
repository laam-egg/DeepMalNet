from .nn import DeepMalNetNNModule, DeepMalNet_Mode
from .ember2024 import NUM_FEATURES

class InferenceDeepMalNetModel:
    def __init__(self) -> None:
        self.model = DeepMalNetNNModule(NUM_FEATURES, DeepMalNet_Mode.INFERENCE)
