import csv
from os.path import join

from utils import bert_tokenizer, dictify

CHOICE_TOKEN_TO_ID = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def process_mctest(input_dir, dataset, file):
    raw_data_file = join(input_dir, dataset, file + '.tsv')
    raw_answer_file = join(input_dir, dataset, file + '.ans')
    processed_lines = []

    with open(raw_data_file) as dataf, open(raw_answer_file) as answerf:
        dataf = csv.reader(dataf, delimiter='\t')
        answerf = csv.reader(answerf, delimiter='\t')

        for data_row, answer_row in zip(dataf, answerf):
            context = data_row[2].replace('\\newline', ' ')
            if len(bert_tokenizer.tokenize(context)) > 475:
                continue

            for question_num in range(len(answer_row)):
                question = data_row[3 + 5 * question_num].replace('multiple:',
                                                                  '').replace(
                    'one:', '')
                choices = data_row[4 + 5 * question_num: 8 + 5 * question_num]
                correct_choice = CHOICE_TOKEN_TO_ID[answer_row[question_num]]

                correct_answers = [choices[correct_choice]]
                incorrect_answers = [c for c in choices if
                                     c != correct_answers[0]]

                processed_lines.append(
                    dictify(context, question, correct_answers,
                            incorrect_answers, dataset)
                )

    return processed_lines