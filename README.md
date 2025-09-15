# DeepMalNet Implementation

- [DeepMalNet Implementation](#deepmalnet-implementation)
  - [Credit](#credit)
  - [Setup](#setup)
  - [Feature Extraction](#feature-extraction)
    - [Extract from Custom PE Files](#extract-from-custom-pe-files)
    - [Use the EMBER2024 Dataset](#use-the-ember2024-dataset)
  - [Split Datasets](#split-datasets)
    - [Pre-splits](#pre-splits)
    - [Split it yourself](#split-it-yourself)
  - [Training](#training)
    - [On Kaggle](#on-kaggle)
    - [On Local Machine](#on-local-machine)
  - [Quick Inference](#quick-inference)
  - [Mass Inference and Evaluation](#mass-inference-and-evaluation)

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
containing the feature vectors of all
samples you specified.

You then have to **rescale the dataset**,
i.e. rescaling the feature vectors to
normalize them, which is essential for
the robustness of the training process.

First, compute the scaling hyperparameters:

```sh
conda activate DeepMalNet
cd $PROJECT_ROOT

python lmdb/compute-scaling-hyperparameters.py \ 
    $PATH_TO_THE_LMDB \ 
    $PROJECT_ROOT/DeepMalNet/models/DeepMalNetModel/hyperparams/
```

Note that in the command above we had to output the
params to

    $PROJECT_ROOT/DeepMalNet/models/DeepMalNetModel/hyperparams/

so that the same hyperparameters could be applied
on inference after training.

Now, rescale the dataset according to these
hyperparameters:

```sh
conda activate DeepMalNet
cd $PROJECT_ROOT

python lmdb/rescale.py \ 
    $PATH_TO_THE_LMDB \ 
    $PATH_TO_THE_NEW_RESCALED_LMDB \ 
    $PROJECT_ROOT/DeepMalNet/models/DeepMalNetModel/hyperparams/
```

**Use the new (rescaled) LMDB** for later
phases e.g. splitting and training.

### Use the EMBER2024 Dataset

The dataset comes in `jsonl` format,
so you need to convert them to feature
vectors.

In fact I have done the conversion
and uploaded the LMDB database to
Kaggle at <https://www.kaggle.com/datasets/laamegg/ember2024-lmdb>.

Following are the steps in case you
want to do it yourself.

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

This process is time-consuming.
It took 2 hours 41 mins to complete
on my Intel i5-8500 CPU.

When it finishes, you have a *LMDB database*
containing the feature vectors of all samples.
You then need to **rescale** the whole dataset
by following the instructions in [the previous section](#extract-from-custom-pe-files).

## Split Datasets

The dataset resulted from the
above feature extraction procedure
must be split into train, CV and
test subsets.

You wouldn't want to load the
whole dataset to memory and split
it there, since the dataset is
huge and three dozens of GiB
worth of RAM (or maybe more)
is needed to hold that much
data.

### Pre-splits

If you used the EMBER2024 dataset
in the feature extraction procedure,
now you have an LMDB database containing
the vectorized EMBER2024 dataset.

You could use the pre-splits
in the given Kaggle dataset
(under the `splits` directory).
Which means, you essentially don't
need to do anything further, since
the splits are already there in
their expected place.

The distribution of the splits:

    train_keys.txt      51.41% zero, 48.59% one (80% overall)
    cv_keys.txt         44.42% zero, 55.58% one (10% overall)
    test_keys.txt       43.19% zero, 56.81% one (10% overall)

### Split it yourself

If you used a custom dataset
in the feature extraction procedure,
then you need to split the LMDB
database yourself.

You could use the script under
`$PROJECT_ROOT/lmdb/split.py`
to split an LMDB database into
multiple parts. For example,
if you want 80% train + 10% CV + 10% test:

```sh
conda activate DeepMalNet
# or a more minimal but still compatible
# venv/conda environment - see the docs
# inside the script for information

cd $PROJECT_ROOT
python ./lmdb/split.py /path/to/lmdb/dir /path/to/splits/output/dir 0.8 0.1 0.1
```

It then outputs 3 files to the specified
splits output directory, each of which
contains the LMDB keys of the corresponding
splits. Read the docs inside the script
for information. In the end, **remember to**
**rename the files** from

    p0.txt
    p1.txt
    p2.txt

to

    train_keys.txt
    cv_keys.txt
    test_keys.txt

and move them to the directory

    /<LMDB_DIR>/splits/

so that we could use it in the Training
phase below.

To view the distribution of each of
those splits (i.e. the number of
zero- and one-labelled samples in
each of the splits):

```sh
conda activate DeepMalNet
# or a more minimal but still compatible
# venv/conda environment - see the docs
# inside the script for information

cd $PROJECT_ROOT
python ./lmdb/split-distribution.py /path/to/lmdb/dir /path/to/splits/output/dir
```

## Training

### On Kaggle

I have uploaded the converted LMDB
database (from EMBER2024 dataset) to Kaggle - link
is [in the previous section](#use-the-ember2024-dataset).
You can run the notebook in

    $PROJECT_ROOT/kaggle/train-on-kaggle.ipynb

on Kaggle with that dataset mounted in,
to train the model. I have also uploaded
and run it here myself: <https://www.kaggle.com/code/laamegg/train-on-kaggle>.

If you have a custom dataset or custom
splits, be sure to upload and mount them
correctly, i.e. following the same directory
structure as that Kaggle dataset I've uploaded.

### On Local Machine

```sh
conda activate DeepMalNet
cd $PROJECT_ROOT

python -m DeepMalNet train /path/to/train/dataset/lmdb/dir
```

Trained DNN checkpoints will be saved in
the directory `$PROJECT_ROOT/checkpoints`.
The checkpoint file is named after the
current timestamp and the number of trained
epochs, to avoid conflicts.

**TODO: pretrained checkpoints for others to experiment with ???**

## Quick Inference

If you want to test the model quickly
on a file or all files under a
directory (scanned recursively):

```sh
conda activate DeepMalNet

python -m DeepMalNet infer /path/to/a/model/checkpoint /path/to/file/or/dir/to/scan
```

The results will be printed directly
to the console.

## Mass Inference and Evaluation

For inference, we use another LMDB as a test set.
You also need to rescale like this:

```sh
conda activate DeepMalNet
cd $PROJECT_ROOT

python lmdb/rescale.py \ 
    $PATH_TO_THE_TEST_LMDB \ 
    $PATH_TO_THE_NEW_RESCALED_TEST_LMDB \ 
    $PROJECT_ROOT/DeepMalNet/models/DeepMalNetModel/hyperparams/
```

which means, you must use the same scaling
hyperparameters as that used in training,
while providing the test LMDB instead.
The new, rescaled test LMDB will be used
hereafter.

[`pefe-ief`](https://github.com/pefe-system/pefe-ief)
is a Python package that supports evaluation of
PE malware detection models. It has been plugged
into this project.

You can run `pefe-ief` to evaluate and compare
several DeepMalNet model checkpoints' performance
on a specified test set (LMDB) with the following
command:

```sh
conda activate DeepMalNet
cd $PROJECT_ROOT

python -m DeepMalNet ief \ 
    /path/to/dir/containing/DeepMalNet/checkpoints/i.e.,/those/trained/.pth/files \ 
    /path/to/rescaled/test/LMDB/dir \ 
    /path/to/dir/containing/evaluation/results
```

where `/path/to/dir/containing/evaluation/results` is
the path to an existing empty directory to contain
the evaluation results when the command completes.

With that evaluation results directory (hereafter
"RESULTS directory" or "results_dir"), follow the
[instructions in the `README` file of `pefe-ief-viz`](https://github.com/pefe-system/pefe-ief-viz)
to visualize the results.

By the way, I have run evaluation of the pretrained
models of mine against my own test set. You could
view the results directly instead of running this
whole IEF thing. They are located in `$PROJECT_ROOT/DeepMalNet_evaluation_results`.
Note that **you have to download the HTML file**
and open it in order for it to display correctly.
Opening it online risks losing interactive
elements (I don't know why!).

Note that you still need to follow `pefe-ief-viz`
instructions if you want to run the notebook
yourself and/or export it to HTML.
