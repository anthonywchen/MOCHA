import glob
import itertools
import json
from os import makedirs
from os.path import isdir, join
import random
from tqdm import tqdm

import jsonlines

from process_boolq import process_boolq
from process_mctest import process_mctest
from process_multirc import process_multirc
from process_race import process_race
from utils import *

random.seed(0)

INPUT_DIR = 'data/pretraining/raw'
OUTPUT_DIR = 'data/pretraining'


def write_processed_lines(processed_lines, output_file):
    with open(output_file, 'w', encoding='utf-8') as writer:
        for line in processed_lines:
            writer.write(json.dumps(line, ensure_ascii=False) + '\n')


def boolq():
    dataset = 'boolq'
    print('Processing', dataset)

    output_dir = join(OUTPUT_DIR, dataset)
    if not isdir(output_dir):
        makedirs(output_dir)

    train_lines = process_boolq(INPUT_DIR, dataset, 'train.jsonl')
    dev_lines = process_boolq(INPUT_DIR, dataset, 'dev.jsonl')
    write_processed_lines(train_lines, join(output_dir, 'train.jsonl'))
    write_processed_lines(dev_lines, join(output_dir, 'dev.jsonl'))


def mctest():
    dataset = 'mctest'
    print('Processing', dataset)

    output_dir = join(OUTPUT_DIR, dataset)
    if not isdir(output_dir):
        makedirs(output_dir)

    train_lines = process_mctest(INPUT_DIR, dataset, 'mc160.train')
    train_lines += process_mctest(INPUT_DIR, dataset, 'mc500.train')
    dev_lines = process_mctest(INPUT_DIR, dataset, 'mc160.dev')
    dev_lines += process_mctest(INPUT_DIR, dataset, 'mc500.dev')
    write_processed_lines(train_lines, join(output_dir, 'train.jsonl'))
    write_processed_lines(dev_lines, join(output_dir, 'dev.jsonl'))


def multirc():
    dataset = 'multirc'
    print('Processing', dataset)

    output_dir = join(OUTPUT_DIR, dataset)
    if not isdir(output_dir):
        makedirs(output_dir)

    train_lines = process_multirc(INPUT_DIR, dataset, 'train_456-fixedIds.json')
    dev_lines = process_multirc(INPUT_DIR, dataset, 'dev_83-fixedIds.json')
    write_processed_lines(train_lines, join(output_dir, 'train.jsonl'))
    write_processed_lines(dev_lines, join(output_dir, 'dev.jsonl'))


def race():
    dataset = 'race'
    print('Processing', dataset)

    output_dir = join(OUTPUT_DIR, dataset)
    if not isdir(output_dir):
        makedirs(output_dir)

    train_lines = process_race(INPUT_DIR, dataset, 'train')
    dev_lines = process_race(INPUT_DIR, dataset, 'dev')
    write_processed_lines(train_lines, join(output_dir, 'train.jsonl'))
    write_processed_lines(dev_lines, join(output_dir, 'dev.jsonl'))


def merge_all():
    train_files = glob.glob(join(OUTPUT_DIR, '*/train.jsonl'))
    dev_files = glob.glob(join(OUTPUT_DIR, '*/dev.jsonl'))

    train_lines = []
    for file in train_files:
        for line in jsonlines.open(file):
            train_lines.append(line)

    dev_lines = []
    for file in dev_files:
        for line in jsonlines.open(file):
            dev_lines.append(line)

    return train_lines, dev_lines


def format_and_augment(lines, augment, output_file):
    ####################
    ### Reformat the merged file so each line has one reference and
    ### one candidate and a label indicating if they are the same.
    ####################
    formatted_lines = []
    print('Formatting...')
    for line in tqdm(lines):
        context = line['context']
        question = line['question']
        correct_answers = line['correct_answers']
        incorrect_answers = line['incorrect_answers']
        dataset = line['dataset']

        # Apply identity augmentation 20% of the time
        for ans in correct_answers:
            if random.random() < 0.2:
                formatted_lines.append({
                    'context': context,
                    'question': question,
                    'reference': ans,
                    'candidate': ans,
                    'dataset': dataset,
                    'same': True
                })

        # Create entries for pairs of correct answer
        for ans1, ans2 in list(itertools.combinations(correct_answers, 2)):
            formatted_lines.append({
                'context': context,
                'question': question,
                'reference': ans1,
                'candidate': ans2,
                'dataset': dataset,
                'same': True
            })

        # Create entries for a pair of a correct and an incorect answer
        for ans1, ans2 in list(
                itertools.product(correct_answers, incorrect_answers)):
            formatted_lines.append({
                'context': context,
                'question': question,
                'reference': ans1,
                'candidate': ans2,
                'dataset': dataset,
                'same': False
            })

    ####################
    ### Write out the reformatted stuff
    ####################
    write_processed_lines(formatted_lines, output_file)


def main():
    boolq()
    mctest()
    multirc()
    race()
    train_lines, dev_lines = merge_all()
    format_and_augment(train_lines, True, join(OUTPUT_DIR, 'train.jsonl'))
    format_and_augment(dev_lines, True, join(OUTPUT_DIR, 'dev.jsonl'))


if __name__ == '__main__':
    main()