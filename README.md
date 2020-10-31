# MOCHA
This repository contains the data and code for the paper `MOCHA: A Dataset for Training and Evaluating Reading Comprehension Metrics.`

Paper: https://arxiv.org/abs/2010.03636

Website: https://allennlp.org/mocha

## Setting up the environment
```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Baseline n-gram metrics
pip install git+https://github.com/salaniz/pycocoevalcap
```

## Getting MOCHA
The MOCHA dataset is available at `data/mocha.tar.gz`. 

To get statistics of MOCHA, first uncompress MOCHA, then run: `python print_mocha_statistics.py`.

To check the SHA-1 hash of a data file, run `sha1sum [filename]` and compare against file `[filename].sha1`. 


## Baseline Metrics
Baseline metrics are BLEU-1, METEOR, ROUGE-L, and BERTScore.

Predictions on dev/test set of MOCHA: `python baseline_metrics/write_mocha_preds.py`.

Predictions on the minimal pairs: `python baseline_metrics/write_minimal_pair_preds.py`

Predictions are written out into `baseline_metrics/preds/`.

## Training LERC
Coming soon.


## Evaluating Predictions
We provide two evaluation scripts, one for evaluating predictions on the core MOCHA set and one for evaluating predictions on the set of minimal pairs.


#### MOCHA evaluation
Example for running evaluation on the dev set of MOCHA set with METEOR predictions:
```
python evaluate_mocha_preds.py 
    --annotations data/mocha/dev.json 
    --predictions baseline_metrics/preds/meteor/dev_preds.json
```
This outputs a file `baseline_metrics/preds/meteor/dev_preds.json.corrs`.

We provide Pearson correlation results for each constituent dataset (`overall` key) as well as for each candidate generation source for each constituent dataset:

```
>>> cat baseline_metrics/preds/meteor/dev_preds.json.corrs
{
    ...
    "drop": {
        "overall": 0.6623693176399084,
        "nabert": 0.5838373885222617,
        "naqanet": 0.7208649541039371
    },
    "mcscript": {
        "overall": 0.4610524843217756,
        "gpt2": 0.4397690870601691,
        "mhpg": 0.6284184707193933,
        "backtranslation": 0.4140887684184332
    },
    ...
}
``` 

#### Minimal pairs evaluation
Example for running evaluation on minimal pairs set with METEOR predictions:
```
python evaluate_minimal_pair_preds.py 
    --annotations data/mocha/minimal_pairs.json
    --predictions baseline_metrics/preds/meteor/minimal_pair_preds.json
```
This outputs a file `baseline_metrics/preds/meteor/minimal_pair_preds.json.score`.

```
>>> cat baseline_metrics/preds/meteor/minimal_pair_preds.json.score
{
    "cosmosqa": 57.0,
    "mcscript": 62.0,
    "narrativeqa": 60.0,
    "socialiqa": 66.0
}
``` 