"""
Template from README of

https://github.com/pefe-system/pefe-ief
"""

import os

def ief(model_checkpoints_dir, test_lmdb_dir_path, results_dir):
    # type: (str, str, str) -> None

    from pefe_ief import IEF
    from pefe_ief.models.abstract_model import AbstractModel
    from pefe_ief.dataset import PEFELMDBDataset
    from typing import Type

    # 0/6. Import your wrapped models
    from ..models.DeepMalNetModel import InferenceDeepMalNetModel

    # 1/6. Specify a directory to contain IEF results
    ief = IEF(results_dir)

    # 2/6. Specify the directories that contain
    #      the model files e.g. checkpoints, weights etc.
    #      Here, two directories containing model checkpoints
    #      of two different model types. For each eligible
    #      file in a directory, the corresponding model
    #      class e.g. SOREL20M_LGBM will be instantiated
    #      and your do_load() method will be called with
    #      the file's absolute path as argument.
    models_dirs_and_classes = {
       model_checkpoints_dir: InferenceDeepMalNetModel,
    } # type:  dict[str, Type[AbstractModel]]

    # 3/6. However, for each file recursively found in those
    #      directories, this function will be called to
    #      determine whether that file is indeed a model
    #      checkpoint file. An example of how it would be
    #      called:
    #
    #      for each file in those directories, recursive iterations:
    #           file_path = path of the file
    #           if not is_model_checkpoint_file(SOREL20M_FFNN, file_path):
    #               continue # skip file
    #           else:
    #               model = model_class()
    #               model.load(file_path)
    #               # ...
    #
    #      The function shall return True if the given
    #      file_path is indeed a model checkpoint file,
    #      and False otherwise.
    def is_model_checkpoint_file(model_class, file_path):
        # type: (Type[AbstractModel], str) -> bool
        return file_path.endswith(".pth")

    # 4/6. To identify models in the generated reports and
    #      visualizations, this function would be called to
    #      get the models' names displayed in those, given
    #      the model_class, checkpoint_path (same as 3/6)
    #      and also model_type_name, which is actually just
    #      `model_class.__name__`.
    def get_model_checkpoint_name(model_class, model_type_name, checkpoint_path):
        # type: (Type[AbstractModel], str, str) -> str | None
        return os.path.splitext(os.path.basename(checkpoint_path))[0] # simple implementation: just get the file name

    # 5/6. Load your test set. You could read X_test and
    #      y_test from LMDB using pefe-ief's PEFELMDBDataset
    #      like this, or you could read from anywhere
    #      else. X_test shall be a 2D ndarray (m rows, n cols)
    #      while Y_test shall be a 1D ndarray (m rows), where m
    #      is the number of test samples, n is the number of
    #      features aka dimensionality of the input feature vector.
    X_test, y_test = PEFELMDBDataset().read(test_lmdb_dir_path)

    # 6/6. Finally, run pefe-ief. The results will be generated in
    #      the directory you specified at 1/6.
    ief.run(
        models_dirs_and_classes=models_dirs_and_classes,
        is_model_checkpoint_file=is_model_checkpoint_file,
        get_model_checkpoint_name=get_model_checkpoint_name,
        X_test=X_test, y_test=y_test,
        config=IEF.EvaluationConfig(thresholds=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9]),
    )
