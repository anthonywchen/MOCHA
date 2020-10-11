import json
import spacy
from collections import defaultdict
from statistics import mean
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner'])


### Count statistics
def num_passages(data):
    seen_passages = set()
    num = 0

    for instance in data.values():
        if instance['context'] not in seen_passages:
            num += 1
            seen_passages.add(instance['context'])

    return num


def num_ques_ref_pairs(data):
    seen_ques_ref_pairs = set()
    num = 0

    for instance in data.values():
        ques_ref = instance['context'] + instance['question'] + instance['reference']
        if ques_ref not in seen_ques_ref_pairs:
            num += 1
            seen_ques_ref_pairs.add(ques_ref)

    return num


def num_instances(data):
    return len(data)


### Average length statistics
def avg_passage_len(data):
    lengths = [len(nlp(instance['context'])) for instance in data.values()]
    return round(mean(lengths), 1)


def avg_question_len(data):
    lengths = [len(nlp(instance['question'])) for instance in data.values()]
    return round(mean(lengths), 1)


def avg_reference_len(data):
    lengths = [len(nlp(instance['reference'])) for instance in data.values()]
    return round(mean(lengths), 1)


def avg_candidate_len(data):
    lengths = [len(nlp(instance['candidate'])) for instance in data.values()]
    return round(mean(lengths), 1)


def get_statistics_for_split(file_path, compute_average_lengths=False):
    data = json.load(open(file_path))
    statistics = defaultdict(lambda: defaultdict(int))

    # Compute statistics per constituent dataset
    for dataset in tqdm(data):
        # Compute count statistics
        statistics[dataset]['num_passages'] = num_passages(data[dataset])
        statistics[dataset]['num_ques_ref_pairs'] = num_ques_ref_pairs(data[dataset])
        statistics[dataset]['num_instances'] = num_instances(data[dataset])

        # Add count statistics to a total field
        statistics['total']['num_passages'] += \
            statistics[dataset]['num_passages']
        statistics['total']['num_ques_ref_pairs'] += \
            statistics[dataset]['num_ques_ref_pairs']
        statistics['total']['num_instances'] += \
            statistics[dataset]['num_instances']

        # Compute average length statistics
        if compute_average_lengths:
            statistics[dataset]['avg_passage_len'] = \
                avg_passage_len(data[dataset])
            statistics[dataset]['avg_question_len'] = \
                avg_question_len(data[dataset])
            statistics[dataset]['avg_reference_len'] = \
                avg_reference_len(data[dataset])
            statistics[dataset]['avg_candidate_len'] = \
                avg_candidate_len(data[dataset])

    return statistics


def main():
    statistics = {
        'train': get_statistics_for_split('data/mocha/train.json',
                                          compute_average_lengths=True),
        'dev': get_statistics_for_split('data/mocha/dev.json'),
        'test': get_statistics_for_split('data/mocha/test_no_labels.json')
    }

    print(json.dumps(statistics, indent=4))


if __name__ == "__main__":
    main()