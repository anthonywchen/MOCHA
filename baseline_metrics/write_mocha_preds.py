""" For each baseline metric, write out MOCHA predictions.

For each of the baseline metrics (BLEU-1, METEOR, ROUGE-L, and BERTScore),
we write out prediction scores on the validation and test set of MOCHA.
"""
import json
import os

from bert_score import score as BERT_SCORE
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu

METEOR = Meteor()
ROUGE = Rouge()
BLEU = Bleu(1)
VALIDATION_DATA_PATH = 'data/mocha/dev.json'
TEST_DATA_PATH = 'data/mocha/test_no_labels.json'
OUTPUT_DIR = 'baseline_metrics/preds'


def write_predictions(predictions, split, metric):
    # Write predictions into `baseline_metrics/preds/<metric_name>/`
    output_dir = os.path.join(OUTPUT_DIR, metric)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_file = split + '_preds.json'
    with open(os.path.join(output_dir, output_file), 'w') as f:
        f.write(json.dumps(predictions, indent=4, sort_keys=True))


def remove_punc(s):
    return s.replace('?', '').replace('.', '').replace('!', '')


def merge_preds(constituent_dataset, pred_scores):
    """ For each prediction score, we add it to a dictionary with the key
        being the instance ID.
    """
    predictions = {
        instance_id: {'pred_score': pred_score}
        for instance_id, pred_score in zip(constituent_dataset, pred_scores)
    }
    return predictions


def make_bleu1_predictions(mocha_dataset, split):
    # For each constituent dataset, make predictions
    predictions = {}
    for dataset in mocha_dataset:
        refs = {i: [remove_punc(instance['reference'])] for i, instance in
                enumerate(mocha_dataset[dataset].values())}
        cands = {i: [remove_punc(instance['candidate'])] for i, instance in
                 enumerate(mocha_dataset[dataset].values())}
        pred_scores = BLEU.compute_score(refs, cands)[1][0]

        # Merge the prediction score back with the MOCHA instance ID
        predictions[dataset] = merge_preds(mocha_dataset[dataset], pred_scores)

    write_predictions(predictions, split, metric='bleu')


def make_meteor_predictions(mocha_dataset, split):
    # For each constituent dataset, make predictions
    predictions = {}
    for dataset in mocha_dataset:
        refs = {i: [remove_punc(instance['reference'])] for i, instance in
                enumerate(mocha_dataset[dataset].values())}
        cands = {i: [remove_punc(instance['candidate'])] for i, instance in
                 enumerate(mocha_dataset[dataset].values())}
        pred_scores = METEOR.compute_score(refs, cands)[1]

        # Merge the prediction score back with the MOCHA instance ID
        predictions[dataset] = merge_preds(mocha_dataset[dataset], pred_scores)

    write_predictions(predictions, split, metric='meteor')


def make_rouge_predictions(mocha_dataset, split):
    # For each constituent dataset, make predictions
    predictions = {}
    for dataset in mocha_dataset:
        refs = {i: [remove_punc(instance['reference'])] for i, instance in
                enumerate(mocha_dataset[dataset].values())}
        cands = {i: [remove_punc(instance['candidate'])] for i, instance in
                 enumerate(mocha_dataset[dataset].values())}
        pred_scores = ROUGE.compute_score(refs, cands)[1]

        # Merge the prediction score back with the MOCHA instance ID
        predictions[dataset] = merge_preds(mocha_dataset[dataset], pred_scores)

    write_predictions(predictions, split, metric='rouge')


def make_bertscore_predictions(mocha_dataset, split):
    # For each constituent dataset, make predictions
    predictions = {}
    for dataset in mocha_dataset:
        refs = [remove_punc(instance['reference']) for instance in
                mocha_dataset[dataset].values()]
        cands = [remove_punc(instance['candidate']) for instance in
                 mocha_dataset[dataset].values()]
        pred_scores = BERT_SCORE(cands, refs, lang='en')[-1].tolist()

        # Merge the prediction score back with the MOCHA instance ID
        predictions[dataset] = merge_preds(mocha_dataset[dataset], pred_scores)

    write_predictions(predictions, split, metric='bertscore')


def make_mocha_predictions(mocha_file):
    # `split` is either `dev` or `test`
    split = os.path.basename(mocha_file).split('.')[0]
    mocha_dataset = json.load(open(mocha_file))

    make_bleu1_predictions(mocha_dataset, split)
    make_meteor_predictions(mocha_dataset, split)
    make_rouge_predictions(mocha_dataset, split)
    make_bertscore_predictions(mocha_dataset, split)


def main():
    make_mocha_predictions(VALIDATION_DATA_PATH)
    make_mocha_predictions(TEST_DATA_PATH)


if __name__ == '__main__':
    main()
