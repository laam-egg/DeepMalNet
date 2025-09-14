from .DeepMalNetNNModule import DeepMalNetNNModule, DeepMalNet_Mode
from .ember2024 import NUM_FEATURES

from pefe_ief.models.abstract_model import AbstractModel
import thrember
import torch
from copy import deepcopy
import numpy as np

class InferenceDeepMalNetModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self._extractor = thrember.PEFeatureExtractor()
    
    @property
    def extractor(self):
        return self._extractor

    def do_extract_features(self, bytes):
        raw_features = self.extractor.raw_features(bytes)
        feature_vector = self.extractor.process_raw_features(raw_features)
        return feature_vector
    


    def do_load(self, model_path):
        self._model = DeepMalNetNNModule(NUM_FEATURES, DeepMalNet_Mode.INFERENCE)

        try:
            model_data = torch.load(model_path)
        except Exception as e:
            if "Attempting to deserialize object on a CUDA device" in str(e):
                model_data = torch.load(model_path, map_location=torch.device('cpu'))
            else:
                raise

        self._model.load_model_data(model_data)
        self._model.eval()

        DEVICE_NAME = "cuda:0"
        try:
            self._model.to(DEVICE_NAME)
        except Exception as e:
            print(f"Failed to use device {DEVICE_NAME}: {e}")
            print(f"Falling back to CPU-only mode.")
            DEVICE_NAME = "cpu"

        self._DEVICE_NAME = DEVICE_NAME
        print(f"Operating on device: {self._DEVICE_NAME}")

    def do_predict(self, feature_vectors):
        X = torch.Tensor(feature_vectors).to(self._DEVICE_NAME)
        raw_predictions = self._model(X)
        y_probs = self.detach_and_copy_array(raw_predictions)
        return y_probs

    def do_get_batch_size(self):
        return 16384 # occupy about 450MB VRAM on NVIDIA GeForce MX350 (2GB)

    @staticmethod
    def detach_and_copy_array(array):
        """
        From <EMBER2024_project_root/evaluate.py>:

        we do a lot of deepcopy stuff here to avoid a FD "leak" in the dataset generator
        see here: https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
        """
        if isinstance(array, torch.Tensor):
            return deepcopy(array.cpu().detach().numpy()).ravel()
        elif isinstance(array, np.ndarray):
            return deepcopy(array).ravel()
        else:
            raise ValueError("Got array of unknown type {} with value {}".format(type(array), str(array)))
