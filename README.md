# MOCHA
This repository contains the data and code for the paper `MOCHA: A Dataset for Training and Evaluating Reading Comprehension Metrics.`

Paper: https://arxiv.org/abs/2010.03636 \
Website: https://allennlp.org/mocha

## Setting up the environment
A setup script is provided which setups a virtual environment and downloads pre-training data. Just run `./setup.sh`. One thing you will have to do is download the BoolQ dataset into `data/pretraining/raw`. See `setup.sh` for more details. 

## MOCHA
MOCHA is available at `data/mocha.tar.gz`.

To get statistics of MOCHA, run `python print_mocha_statistics.py`. \
To check the hash of a MOCHA file, run `sha1sum [filename]` and compare against `[filename].sha1`. 

## Baseline Metrics
Baseline metrics are BLEU-1, METEOR, ROUGE-L, and BERTScore.

Predictions on dev/test set of MOCHA: `python baseline_metrics/write_mocha_preds.py`. \
Predictions on the minimal pairs: `python baseline_metrics/write_minimal_pair_preds.py`.

Predictions are written out into `baseline_metrics/preds/`.

## Training and Using LERC
Training of LERC heavily relies on `allennlp`. This means that training parameters are all specified through JSON config files in `lerc/`. If training doesn't fit into memory, you can increase the `num_gradient_accumulation_steps` while proportionally decreasing `batch_size` in the config files.


Training of LERC is broken down into pre-training on QA datasets followed by fine-tuning on MOCHA. 

#### Pre-training
First process the raw QA dataset files:
```
python data/pretraining/process_pretraining_data.py
```
This script is pretty messy so try not to read it. This creates two files `data/pretraining/train.jsonl` and `data/pretraining/dev.jsonl`.

Then to pre-train LERC, run: 
```
allennlp train lerc/pretrain_config.json -s out/pretraining --include-package lerc
```
This writes out the pre-trained model files into `out/pretraining`.

#### Fine-tuning
To fine-tune the pre-trained model on MOCHA, run:
```
allennlp train lerc/lerc_config.json -s out/lerc --include-package lerc
```
This writes out the trained model into `out/lerc`.

Note that this LERC model is trained on all constituent datasets (AD setting in the paper). To hold out a constituent dataset(s) from training (OOD setting in the paper), add the name of that dataset into the `holdout_sets` key in `lerc/lerc_config.json`. The names of the six constituent datasets are `cosmosqa, drop, mcscript, narrativeqa, quoref, socialiqa`.

#### Using LERC for Predictions
To use LERC to make individual predictions:
```python
from lerc.lerc_predictor import LERCPredictor

# Loads an AllenNLP Predictor that wraps our trained model
predictor = LERCPredictor.from_path(
    archive_path='out/lerc', # Path to the model archive
    predictor_name='lerc',   # model name
    cuda_device=0            # Set CUDA device with this parameter
)

# The instance we want to get LERC score for in a JSON format
input_json = {
    'context': 'context string',
    'question': 'question string',
    'reference': 'reference string', 
    'candidate': 'candidate string'
}

output_dict = predictor.predict_json(input_json)
print('Predicted LERC Score:', output_dict['pred_score'])
```

#### MOCHA and Minimal Pair Predictions
To get LERC predictions on the validation set of MOCHA:
```
python -m lerc.write_mocha_preds --mocha_file data/mocha/dev.json -s out/lerc
```
To get LERC predictions on the set of set of minimal pairs
```
python -m lerc.write_minimal_pair_preds --mocha_file data/mocha/minimal_pairs.json -s out/lerc
```

The predictions are written out into `out/lerc/preds/`.

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
