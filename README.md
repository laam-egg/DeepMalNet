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
