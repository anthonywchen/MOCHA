import argparse
import json
from collections import defaultdict
from scipy.stats import pearsonr


def correlation(annotations, predictions):
    scores, pred_scores = [], []

    for instance_id in annotations:
        if instance_id not in predictions:
            print('No prediction for %s. Assigning 0.' % instance_id)
            pred_scores.append(0)
        else:
            pred_scores.append(predictions[instance_id]['pred_score'])
        scores.append(annotations[instance_id]['score'])

    return pearsonr(scores, pred_scores)[0]


def get_sources(annotations, predictions):
    sources = set(
        [instance['metadata']['source'] for instance in annotations.values()])

    for source in sources:
        # Get all annotations for the current source
        annotations_for_source = {
            instance_id: instance
            for instance_id, instance in annotations.items() if
            instance['metadata']['source'] == source
        }

        # Get all predictions for the current source
        predictions_for_source = {
            instance_id: predictions[instance_id]
            for instance_id in annotations_for_source if
            instance_id in predictions
        }

        yield source, annotations_for_source, predictions_for_source


def evaluate(annotations, predictions):
    metrics = defaultdict(dict)

    # For each constituent dataset, calculate Pearson correlation
    for dataset in annotations:
        metrics[dataset]['overall'] = correlation(annotations[dataset], predictions[dataset])

        # For each generation source, calculate the correlation for that source
        for source, annotations_for_source, predictions_for_source in \
                get_sources(annotations[dataset], predictions[dataset]):
            metrics[dataset][source] = correlation(annotations_for_source, predictions_for_source)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotations', type=str, required=True)
    parser.add_argument('-p', '--predictions', type=str, required=True)
    args = parser.parse_args()

    annotations = json.load(open(args.annotations))
    predictions = json.load(open(args.predictions))
    metrics = evaluate(annotations, predictions)

    # Write correlation scores
    output_file = args.predictions + '.corrs'
    with open(output_file, 'w') as writer:
        writer.write(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    main()
