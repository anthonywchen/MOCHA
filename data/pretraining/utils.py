import itertools
import math
from nltk.corpus import words
import numpy as np
import spacy
import random
import torch
from typing import Dict
from transformers import BertForMaskedLM, BertTokenizer
from spacy.lang.en.stop_words import STOP_WORDS
import string

CUDA_DEVICE = 1
STOP_WORDS.update(string.punctuation)
nlp = spacy.load('en_core_web_sm', disable='ner')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertForMaskedLM.from_pretrained('bert-base-uncased').to(CUDA_DEVICE)


def dictify(
        context: str,
        question: str,
        correct_answers: list,
        incorrect_answers: list,
        dataset: str
) -> Dict:
    assert type(context) == str
    assert type(question) == str
    assert type(correct_answers) == list
    assert type(incorrect_answers) == list

    d = {
        'context': context,
        'question': question,
        'correct_answers': correct_answers,
        'incorrect_answers': incorrect_answers,
        'dataset': dataset
    }

    return d


################
### Data functions
################
def tokenize_and_filter_stop_words(string):
    return [token.text for token in nlp(string) if token.text not in STOP_WORDS]


def random_word_replacement(answer):
    """
    Randomly replaces tokens in the answer with randomly sampled tokens from a word list (from NLTK)
    """
    answer = answer.split()
    if len(answer) <= 3:
        return

    # Number of words to replace in the answer
    num_words_to_replace = len(answer) // 2
    indices_to_replace = random.sample(range(len(answer)), num_words_to_replace)
    random_words = random.sample(words.words(), num_words_to_replace)

    for pos, index in enumerate(indices_to_replace):
        answer[index] = random_words[pos]
    return ' '.join(answer)


def context_word_replacement(context, answer):
    """
    Randomly replaces tokens in the answer with randomly sampled tokens from the context
    """
    context = context.split()
    answer = answer.split()

    if len(answer) < 4:
        return

    # Number of words to replace in the answer
    num_words_to_replace = len(answer) // 2
    indices_to_replace = random.sample(range(len(answer)), num_words_to_replace)
    random_words = random.sample(context, num_words_to_replace)

    for pos, index in enumerate(indices_to_replace):
        answer[index] = random_words[pos]

    return ' '.join(answer)


def permute_words(answer):
    """
    Permutes an answer such that none of the orignal words are in their original positions
    """
    answer = answer.split()
    if len(answer) < 5:
        return

    random.shuffle(answer)
    all_permuatations = itertools.permutations(answer)

    # Return the first permutation where none of the tokens are in their original positions
    for permutation in all_permuatations:
        for pos in range(len(answer)):
            if permutation[pos] == answer[
                pos]:  # Skip b/c tokens match at positions `pos`
                continue
            return ' '.join(permutation)


def drop_words(answer):
    answer = answer.split()
    if len(answer) < 6:
        return

    # Keep 50% of the words
    num_tokens_to_keep = int(len(answer) * .5)
    indices_to_keep = sorted(
        random.sample(range(len(answer)), num_tokens_to_keep))

    # Drop words in tokenized answer
    answer = [w for pos, w in enumerate(answer) if pos in indices_to_keep]

    return ' '.join(answer)


def masked_fillin(answer):
    tokenized_answer = bert_tokenizer.tokenize(answer)
    if len(tokenized_answer) < 6:
        return

    tokens_to_mask = math.ceil(len(tokenized_answer) * .4)
    indices_to_replace = sorted(
        random.sample(range(len(tokenized_answer)), tokens_to_mask))

    masked_input = [token if pos not in indices_to_replace else '[MASK]' for
                    pos, token in enumerate(tokenized_answer)]
    masked_input = bert_tokenizer.encode(masked_input, add_special_tokens=True)

    input_ids = torch.tensor(masked_input).unsqueeze(0).to(CUDA_DEVICE)
    logits = bert(input_ids)[0]

    for pos in indices_to_replace:
        probs = torch.nn.functional.softmax(
            logits[0][pos + 1]).cpu().detach().numpy()
        probs /= probs.sum()
        # Sample a word to fill in a MASk
        guess = np.random.choice(range(bert_tokenizer.vocab_size), 1, p=probs)
        guess_token = bert_tokenizer.convert_ids_to_tokens([guess.item()])
        tokenized_answer[pos] = guess_token[0]

    return ' '.join([x for x in tokenized_answer]).replace(' ##', '')


def are_two_references_the_same(question, ref_ans, cand_ans):
    """
    Used to determine if two answers (from the same question) are the same via n-gram overlap.
    This is useful for cases like MultiRC where there are multiple correct answers but
    it isn't clear whether they are the same.
    """
    stripped_question = [token.text for token in nlp(question) if
                         token.text not in STOP_WORDS]
    stripped_ref_ans = [token.text for token in nlp(ref_ans) if
                        token.text not in STOP_WORDS]
    stripped_cand_ans = [token.text for token in nlp(cand_ans) if
                         token.text not in STOP_WORDS]

    # Get overlap between ans1 and ans2
    token_overlap = len(set(stripped_ref_ans) & set(stripped_cand_ans))
    shorter_ans_len = min(len(stripped_ref_ans), len(stripped_cand_ans))

    if shorter_ans_len == 0 or token_overlap / shorter_ans_len >= 0.5:
        return True

    return False