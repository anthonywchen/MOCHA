import glob
import json
from os.path import join
from tqdm import tqdm

from utils import bert_tokenizer, dictify

CHOICE_TOKEN_TO_ID = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def process_race(input_dir, dataset, data_type):
    raw_data_dir = join(input_dir, dataset, data_type)
    raw_data_files = glob.glob(join(raw_data_dir, '*/*'))
    processed_lines = []

    for file in tqdm(raw_data_files):
        raw_data = json.load(open(file))

        context = raw_data['article']
        if len(bert_tokenizer.tokenize(context)) > 475:
            continue

        # THere are a couple of questions per context
        for i in range(len(raw_data['answers'])):
            question = raw_data['questions'][i]

            # Skip cloze style questions
            if '?' not in question:
                continue

            choices = raw_data['options'][i]
            correct_choice = CHOICE_TOKEN_TO_ID[raw_data['answers'][i]]

            correct_answers = [choices[correct_choice]]
            incorrect_answers = [c for c in choices if c != correct_answers[0]][
                                :2]

            if len(incorrect_answers) < 2:
                continue

            processed_lines.append(
                dictify(context, question, correct_answers, incorrect_answers,
                        dataset)
            )

    return processed_lines