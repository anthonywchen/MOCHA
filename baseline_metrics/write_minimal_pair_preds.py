""" For each baseline metric, write out minimal pair predictions.

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
DATA_PATH = 'data/mocha/minimal_pairs.json'
OUTPUT_DIR = 'baseline_metrics/preds'


def write_predictions(predictions, metric):
    # Write predictions into `baseline_metrics/preds/<metric_name>/`
    output_dir = os.path.join(OUTPUT_DIR, metric)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, 'minimal_pair_preds.json'), 'w') as f:
        f.write(json.dumps(predictions, indent=4, sort_keys=True))


def remove_punc(s):
    return s.replace('?', '').replace('.', '').replace('!', '')


def merge_preds(constituent_dataset, pred_scores1, pred_scores2):
    """ For each prediction score, we add it to a dictionary with the key
        being the instance ID.
    """
    predictions = {
        instance_id: {'pred_score1': pred_score1, 'pred_score2': pred_score2}
        for instance_id, pred_score1, pred_score2 in
        zip(constituent_dataset, pred_scores1, pred_scores2)
    }
    return predictions


def make_bleu1_predictions(mp_dataset):
    # For each dataset, make predictions
    predictions = {}
    for dataset in mp_dataset:
        refs = {i: [remove_punc(instance['reference'])] for i, instance in
                enumerate(mp_dataset[dataset].values())}
        cands1 = {i: [remove_punc(instance['candidate1'])] for i, instance in
                  enumerate(mp_dataset[dataset].values())}
        cands2 = {i: [remove_punc(instance['candidate2'])] for i, instance in
                  enumerate(mp_dataset[dataset].values())}

        # Compute metric scores for candidate 1 and candidate 2
        pred_scores1 = BLEU.compute_score(refs, cands1)[1][0]
        pred_scores2 = BLEU.compute_score(refs, cands2)[1][0]

        # Merge predictions with instance IDs
        predictions[dataset] = \
            merge_preds(mp_dataset[dataset], pred_scores1, pred_scores2)

    write_predictions(predictions, metric='bleu')


def make_meteor_predictions(mp_dataset):
    # For each dataset, make predictions
    predictions = {}
    for dataset in mp_dataset:
        refs = {i: [remove_punc(instance['reference'])] for i, instance in
                enumerate(mp_dataset[dataset].values())}
        cands1 = {i: [remove_punc(instance['candidate1'])] for i, instance in
                  enumerate(mp_dataset[dataset].values())}
        cands2 = {i: [remove_punc(instance['candidate2'])] for i, instance in
                  enumerate(mp_dataset[dataset].values())}

        # Compute metric scores for candidate 1 and candidate 2
        pred_scores1 = METEOR.compute_score(refs, cands1)[1]
        pred_scores2 = METEOR.compute_score(refs, cands2)[1]

        # Merge predictions with instance IDs
        predictions[dataset] = \
            merge_preds(mp_dataset[dataset], pred_scores1, pred_scores2)

    write_predictions(predictions, metric='meteor')


def make_rouge_predictions(mp_dataset):
    # For each dataset, make predictions
    predictions = {}
    for dataset in mp_dataset:
        refs = {i: [remove_punc(instance['reference'])] for i, instance in
                enumerate(mp_dataset[dataset].values())}
        cands1 = {i: [remove_punc(instance['candidate1'])] for i, instance in
                  enumerate(mp_dataset[dataset].values())}
        cands2 = {i: [remove_punc(instance['candidate2'])] for i, instance in
                  enumerate(mp_dataset[dataset].values())}

        # Compute metric scores for candidate 1 and candidate 2
        pred_scores1 = ROUGE.compute_score(refs, cands1)[1]
        pred_scores2 = ROUGE.compute_score(refs, cands2)[1]

        # Merge predictions with instance IDs
        predictions[dataset] = \
            merge_preds(mp_dataset[dataset], pred_scores1, pred_scores2)

    write_predictions(predictions, metric='rouge')


def make_bert_score_predictions(mp_dataset):
    # For each dataset, make predictions
    predictions = {}
    for dataset in mp_dataset:
        refs = [remove_punc(instance['reference']) for instance in
                mp_dataset[dataset].values()]
        cands1 = [remove_punc(instance['candidate1']) for instance in
                  mp_dataset[dataset].values()]
        cands2 = [remove_punc(instance['candidate2']) for instance in
                  mp_dataset[dataset].values()]

        # Compute metric scores for candidate 1 and candidate 2
        pred_scores1 = BERT_SCORE(cands1, refs, lang='en')[-1].tolist()
        pred_scores2 = BERT_SCORE(cands2, refs, lang='en')[-1].tolist()

        # Merge predictions with instance IDs
        predictions[dataset] = \
            merge_preds(mp_dataset[dataset], pred_scores1, pred_scores2)

    write_predictions(predictions, metric='bertscore')


def main():
    mp_dataset = json.load(open(DATA_PATH))
    make_bleu1_predictions(mp_dataset)
    make_meteor_predictions(mp_dataset)
    make_rouge_predictions(mp_dataset)
    make_bert_score_predictions(mp_dataset)


if __name__ == '__main__':
    main()
