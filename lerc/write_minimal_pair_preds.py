import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm

from lerc.lerc_predictor import LERCPredictor


def write_predictions(predictions, serialization_dir):
    output_dir = os.path.join(serialization_dir, 'preds')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'minimal_pair_preds.json'), 'w') as f:
        f.write(json.dumps(predictions, indent=4, sort_keys=True))


def make_predictions(minimal_pairs_file, serialization_dir, device):
    predictor = LERCPredictor.from_path(serialization_dir, 'lerc', device)
    predictions = defaultdict(dict)

    mp_dataset = json.load(open(minimal_pairs_file))
    for dataset in mp_dataset:
        for instance_id, line in tqdm(mp_dataset[dataset].items()):
            output_dict1 = predictor.predict_json({
                'context': line['context'],
                'question': line['question'],
                'reference': line['reference'],
                'candidate': line['candidate1']
            })
            output_dict2 = predictor.predict_json({
                'context': line['context'],
                'question': line['question'],
                'reference': line['reference'],
                'candidate': line['candidate2']
            })

            predictions[dataset][instance_id] = {
                'pred_score1': output_dict1['pred_score'],
                'pred_score2': output_dict2['pred_score']
            }

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minimal_pair_file', type=str, required=True,
                        help='serialization directory containing trained model')
    parser.add_argument('-s', '--serialization_dir', type=str, required=True,
                        help='serialization directory containing trained model')
    parser.add_argument('-d', '--device', default=-1, type=int,
                        help='GPU to use. Default is -1 for CPU')
    args = parser.parse_args()

    predictions = make_predictions(args.minimal_pair_file, args.serialization_dir, args.device)
    write_predictions(predictions, args.serialization_dir)


if __name__ == '__main__':
    main()
