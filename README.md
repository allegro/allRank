# allRank : Learning to Rank in PyTorch

### About

allRank is a framework for training learning-to-rank neural models. 
It is based on PyTorch and implements a few common losses used in Learning-to-rank.

Using allRank you can experiment with your own loss for LTR, and with various Neural architectures. 
Fully-Connected architecture and Transformer architecture are already implemented.

## Motivation

allRank provides an easy and flexible way to experiment with various neural network models and losses.
It is easy to add a custom loss, and to configure model and training procedure.  

## Features

### Implemented losses:
 1. ListNet (For a binary and graded relevance)
 2. ListMLE
 3. RankNet
 4. Ordinal loss
 5. LambdaRank
 6. LambdaLoss
 7. ApproxNDCG
 8. RMSE

## Usage & config

We have prepared an example config that allows to train a small model.

```config.json```

You need to prepare a config file with location of your dataset in libsvm format and model and then run:

```python allrank/main.py --config_file_name allrank/config.json --run_id <the_name_of_your_experiment> --output <the_place_to_save_results>```

### Implementing custom loss

You need to implement a function that takes two tensors as an input and put this function in the `losses` package and make sure it is exposed on a package level.
Next - pass the name (and args if your loss method has some hyperparameters) of your function in a place in a config file:

```
"loss": {
    "name": "yourLoss",
    "args": {}
  }
```

## Continuous integration

You should run `scripts/ci.sh` to verify that code passes style guidelines and unit tests

## License

Apache 2 License
