# allRank MSLR-WEB30K reproducibility guide

## Introduction
In this guide we provide all the necessary information to reproduce allRank results from papers
[Context-Aware Learning to Rank with Self-Attention](https://arxiv.org/abs/2005.10084) and [NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting](https://arxiv.org/abs/2102.07831) on the [MSLR-WEB30K dataset](https://www.microsoft.com/en-us/research/project/mslr/).

## Data preprocessing
The same preprocessing steps were used for both papers. They are performed by the `normalize_features.py` script. Some features are only standardised, while most are log-transformed before standardisation. The script takes three parameters:
* `--ds_path` - path to the MSLR-WEB30K fold to preprocess
* `--features_without_logarithm` - features to standardise without prior log-transform
* `--features_negative` - features that need to be shifted by a constant to ensure strict positivity prior to log-transform

Default values (used in both papers) of `--features_without_logarithm` and `--features_negative` are supplied in the script.
Please note that the script needs to be run on each MSLR-WEB30K fold separately.

## Configuration files
Selected configuration files are supplied in the `configs/` directory. For Context-Aware Learning to Rank with Self-Attention, they are MLP and context-aware rankers with two loss functions:
* ordinal loss, the best-performing context-aware ranker
* NDCGLoss2++, the best-performing MLP ranker

Configuration files for NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting are context-aware rankers trained with loss functions:
* NeuralNDCG@max, the best-performing NeuralNDCG variant
* ApproxNDCG, another direct NDCG optimization method
* LambdaRank@max, the best-performing loss function

Please remember about filling the correct dataset path in the configuration files. These files can be adapted for any other loss function/model combination, requiring only minor changes in model and loss function parameters.

**Important** - in case of any OOM errors on folds 2 and 4 and Context-Aware Learning to Rank configuration files, please try reducing the batch size to 32.
## Notes
The results in the first paper are averaged (+ standard deviation) on validation subsets of five dataset folds while the NeuralNDCG paper supplies a single number, the Fold 1 test subset result.

The whole dataset has ~3% queries with no relevant items resulting in IDCG == 0. Following XGBoost and LightGBM default behaviour, we assume NDCG = 1 for such queries.
The results can be rescaled for other IDCG == 0 treatment methods without the need to re-run any experiment.
For example, one can rescale the result from NDCG = 1 to  NDCG = 0 treatment by applying a simple formula: ```avg_ndcg - (num_blank/num_all)```.
Below is a table with the number of "blank" queries for each fold of the dataset and train/validation/test sets.

|      |   Train   |         |    Test   |         | Validation |         |
|:----:|:---------:|:-------:|:---------:|:-------:|:----------:|---------|
| Fold | num_blank | num_all | num_blank | num_all | num_blank  | num_all |
| 1    | 602       | 18919   | 189       | 6306    | 191        | 6306    |
| 2    | 587       | 18918   | 206       | 6307    | 189        | 6306    |
| 3    | 581       | 18918   | 195       | 6306    | 206        | 6307    |
| 4    | 586       | 18919   | 201       | 6306    | 195        | 6306    |
| 5    | 590       | 18919   | 191       | 6306    | 201        | 6306    |

The configuration files were doublechecked prior to release on GCP AI Platform `n1-highmem16` machines with P100 GPUs.
