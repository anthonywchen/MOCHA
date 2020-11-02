import jsonlines
from os.path import join

from utils import bert_tokenizer, dictify


def process_boolq(input_dir, dataset, raw_data_file):
    raw_data_file = join(input_dir, dataset, raw_data_file)
    processed_lines = []

    for line in jsonlines.open(raw_data_file):
        context = line['passage']
        if len(bert_tokenizer.tokenize(context)) > 475:
            continue

        question = line['question']
        correct_answers = ['yes'] if line['answer'] == True else ['no']
        incorrect_answers = ['yes'] if line['answer'] == False else ['no']

        processed_lines.append(
            dictify(context, question, correct_answers, incorrect_answers,
                    dataset)
        )

    return processed_lines
