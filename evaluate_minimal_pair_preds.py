import argparse
import json


def score_constituent_predictions(annotations, predictions):
    num_correct = 0

    for instance_id in annotations:
        if instance_id not in predictions:
            print('No prediction for %s' % instance_id)
            continue

        annot = annotations[instance_id]
        pred = predictions[instance_id]

        # The first candidate should have a better score than the second candidate
        assert annot['score1'] > annot['score2']

        # If both candidates have the same score, it is half-correct
        if pred['pred_score1'] == pred['pred_score2']:
            num_correct += 0.5
        elif pred['pred_score1'] > pred['pred_score2']:
            num_correct += 1

    return 100*num_correct/len(annotations)


def evaluate(annotations, predictions):
    scores = {
        dataset: score_constituent_predictions(
            annotations[dataset], predictions[dataset]
        )
        for dataset in annotations
    }
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotations', type=str, required=True)
    parser.add_argument('-p', '--predictions', type=str, required=True)
    args = parser.parse_args()

    annotations = json.load(open(args.annotations))
    predictions = json.load(open(args.predictions))
    scores = evaluate(annotations, predictions)

    # Write scores of minimal pair predictions
    output_file = args.predictions + '.score'
    with open(output_file, 'w') as writer:
        writer.write(json.dumps(scores, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
