# DeepMalNet Implementation

## Credit

DeepMalNet is proposed in this paper:

Ravi, Vinayakumar & Kp, Soman. (2018). DeepMalNet: Evaluating shallow and deep networks for static PE malware detection. ICT Express. 4. [10.1016/j.icte.2018.10.006](https://doi.org/10.1016/j.icte.2018.10.006).

They did not provide source code for us.

So I wrote an implementation to test things out.

- This is a binary classifier - benign or malicious.
- They disclosed [the full network architecture here](https://github.com/vinayakumarr/dnn-ember/blob/master/DNN-info.pdf).
- In the paper, they mentioned that the "input layer
    contains 2350 neurons."
- It is likely that the model accepts EMBER features as
    input (though the paper does not directly state so,
    it does mention EMBER in context).
- However, an EMBER2017 or 2018 feature vector has 2381
    features. (The paper is published in 2018). EMBER2024's
    has 2568.
- So there is always some mismatch if we try to use EMBER.
- Therefore, I decided to use all EMBER2024 features anyway,
    which means **the input dim is no longer 2350, but 2568**.
    The rest of the architecture is respected.

## Setup

```sh
conda env create -f environment.yml
# When modifying dependencies:
conda env update -f environment.yml --prune
```

## Feature Extraction

The model uses EMBER2024 extracted features,
hence the input vector consists of
2568 features. The existing code willl
load extracted features from LMDB
databases containing the feature
vectors and labels of the extracted
PE files.

You have to prepare one featurized dataset
for training, and one for evaluation - that
is, *two LMDB databases*.

To build these two databases, you could use
the EMBER2024 dataset, or you could come
up with your own set of files.

### Extract from Custom PE Files

If you want to extract EMBER2024 features
from custom PE files, have a look at
[this `pefe-agent` implementation](https://github.com/laam-egg/EMBER2024?tab=readme-ov-file#mass-feature-extraction)
([what is `pefe-agent`?](https://github.com/pefe-system/pefe-loader)).
After that you will get a *LMDB database*
that this program could read to train
and evaluate the model.

### Use the EMBER2024 Dataset

The dataset comes in `jsonl` format,
so you need to convert them to feature
vectors.

First, download them if you haven't already:

```sh
conda activate DeepMalNet
cd $PROJECT_ROOT

python ./EMBER2024/download_dataset.py
```

This will download the `jsonl` files from
the EMBER2024 dataset that contains extracted
features of PE files only (i.e. not APK or
something else). They will be put in
`$PROJECT_ROOT/dataset/EMBER2024`.

To convert them to vectorized features
for use in DeepMalNet training/evaluation
i.e. convert to a compatible *LMDB database*:

```sh
conda activate DeepMalNet
cd $PROJECT_ROOT

python ./EMBER2024/vectorize_dataset_to_lmdb.py /path/to/lmdb/dir
```

This process may be time-consuming.
It took 2 hours 41 mins to complete
on an Intel i5-8500 CPU.

## Training

```sh
conda activate DeepMalNet
cd $PROJECT_ROOT

python -m DeepMalNet train /path/to/train/dataset/lmdb/dir
```

**TODO: run it???**

Trained DNN checkpoints will be saved in
the directory `$PROJECT_ROOT/checkpoints`.
The checkpoint file is named after the
current timestamp to avoid conflicts.

I also trained some model checkpoints
in that directory.

**TODO: `pefe-ief-viz` ???**

## Quick Inference

**TODO: `@baker.command def infer():` ???**

If you want to test the model quickly
on a file or all files under a
directory (scanned recursively):

```sh
conda activate DeepMalNet

python -m DeepMalNet infer /path/to/file/or/dir
```

The results will be printed directly
to the console.
