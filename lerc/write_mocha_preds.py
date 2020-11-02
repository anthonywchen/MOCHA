import argparse
import json
import os
from collections import defaultdict
from tqdm import tqdm

from lerc.lerc_predictor import LERCPredictor


def write_predictions(predictions, serialization_dir, split):
    # Write out predictions into `serialization_dir/preds/`
    output_dir = os.path.join(serialization_dir, 'preds')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_file = split + '_preds.json'
    with open(os.path.join(output_dir, output_file), 'w') as writer:
        writer.write(json.dumps(predictions, indent=4, sort_keys=True))


def make_predictions(mocha_file, serialization_dir, device):
    predictor = LERCPredictor.from_path(serialization_dir, 'lerc', device)
    predictions = defaultdict(dict)

    mocha_dataset = json.load(open(mocha_file))
    for dataset in mocha_dataset:
        for instance_id, line in tqdm(mocha_dataset[dataset].items()):
            output_dict = predictor.predict_json(line)

            predictions[dataset][instance_id] = {
                'pred_score': output_dict['pred_score']
            }

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mocha_file', type=str, required=True,
                        help='MOCHA file to get predictions on')
    parser.add_argument('-s', '--serialization_dir', type=str, required=True,
                        help='serialization directory containing trained model')
    parser.add_argument('-d', '--device', default=-1, type=int,
                        help='GPU to use. Default is -1 for CPU')
    args = parser.parse_args()

    predictions = make_predictions(args.mocha_file,args.serialization_dir,args.device)

    # `split` is either `dev` or `test`
    split = os.path.basename(args.mocha_file).split('.')[0]
    write_predictions(predictions, args.serialization_dir, split)


if __name__ == '__main__':
    main()
