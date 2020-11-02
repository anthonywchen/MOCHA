import json
from os.path import join
from tqdm import tqdm

from utils import bert_tokenizer, dictify


def process_passage(passage):
    """
        Strips sentences in passages of their sentence number and
        removes the block notation for each sentence.
        Returns a list of the sentences in the passage.
    """
    passage_sents = [sent.split('</b>')[-1] for sent in
                     passage.strip().split('<br>')]
    passage_sents = [sent for sent in passage_sents if sent != '']

    # Check that our passage was parserd correctly and we have right # of sentences
    assert passage.count('<br>') == len(passage_sents)

    return ' '.join(passage_sents)


def process_multirc(input_dir, dataset, raw_data_file):
    raw_data_file = join(input_dir, dataset, raw_data_file)
    raw_data = json.load(open(raw_data_file))
    processed_lines = []

    # Each entry in `raw_data['data']` is a dictionary associated with
    # a passage and its associated questions and answer choices
    for passage_dict in tqdm(raw_data['data']):
        context = process_passage(passage_dict['paragraph']['text'])
        if len(bert_tokenizer.tokenize(context)) > 475:
            continue

        # Iterate through the questions associated with each passage
        for question_dict in passage_dict['paragraph']['questions']:
            question = question_dict['question']
            correct_answers = [answer_dict['text'] for answer_dict in
                               question_dict['answers'] if
                               answer_dict['isAnswer'] == 1]
            incorrect_answers = [answer_dict['text'] for answer_dict in
                                 question_dict['answers'] if
                                 answer_dict['isAnswer'] == 0]

            if correct_answers == [] or incorrect_answers == []:
                continue

            processed_lines.append(
                dictify(context, question, correct_answers, incorrect_answers,
                        dataset)
            )

    return processed_lines