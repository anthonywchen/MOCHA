import logging
import json
import numpy as np
from overrides import overrides
from transformers import BertTokenizer

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("lerc")
class LERCDatasetReader(DatasetReader):
    def __init__(
        self,
        bert_model: str = 'bert-base-uncased',
        max_length: int = 512,
        holdout_sets: list = [],
        augment: bool = True,
        lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self.max_length = max_length
        self.holdout_sets = holdout_sets if type(holdout_sets) == list else [holdout_sets]
        self.augment = augment
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

    @overrides
    def _read(self, file_path: str):
        lines = []
        mocha_dataset = json.load(open(file_path))

        # Check that if we specified datasets to hold out, that they are
        # indeed in the MOCHA dataset.
        for constituent_dataset in self.holdout_sets:
            assert constituent_dataset in mocha_dataset.keys()

        # Iterate through the constituent datasets, loading the MOCHA instances
        for constituent_dataset in mocha_dataset:
            seen_questions = set()
            if constituent_dataset in self.holdout_sets:
                continue

            for line in mocha_dataset[constituent_dataset].values():
                # Append the current instance
                lines.append({
                    'context': line['context'],
                    'question': line['question'],
                    'reference': line['reference'],
                    'candidate': line['candidate'],
                    'score': line['score'],
                })

                # Do a little data augmentation if the flag is set.
                if self.augment:
                    # Identity augmentation with the reference
                    # If this is the first time we have seen the question,
                    # create an identity instance.
                    if line['question'] not in seen_questions:
                        lines.append({
                            'context': line['context'],
                            'question': line['question'],
                            'reference': line['reference'],
                            'candidate': line['reference'],
                            'score': 5,
                        })
                        seen_questions.add(line['question'])

                    # Augmentations via flipping reference and candidate
                    # If the current line has a perfect score, flip the
                    # reference and candidate
                    if self.augment and line['score'] == 5:
                        lines.append({
                            'context': line['context'],
                            'question': line['question'],
                            'reference': line['candidate'],
                            'candidate': line['reference'],
                            'score': 5,
                        })

        # Create instances
        for line in lines:
            yield self.text_to_instance(**line)

    @overrides
    def text_to_instance(
        self, context, question, reference, candidate, score=None
    ) -> Instance:
        context_tokens = self.tokenizer.tokenize(context)
        question_tokens = self.tokenizer.tokenize(question)
        reference_tokens = self.tokenizer.tokenize(reference)
        candidate_tokens = self.tokenizer.tokenize(candidate)

        # Truncates the context if the BERT input would be too long
        context_tokens = self.truncate_context(
            context_tokens, question_tokens, reference_tokens, candidate_tokens
        )

        # Creates the BERT input (input IDs, segment IDs, and attention mask)
        input_ids, token_type_ids, attention_mask = self.create_input(
            context_tokens, question_tokens, reference_tokens, candidate_tokens
        )

        fields = {
            'input_ids': ArrayField(np.array(input_ids), dtype=np.int64,
                                    padding_value=self.tokenizer.pad_token_id),
            'token_type_ids': ArrayField(np.array(token_type_ids),
                                         dtype=np.int64),
            'attention_mask': ArrayField(np.array(attention_mask),
                                         dtype=np.int64),
            'metadata': MetadataField({
                'context': context,
                'context_tokens': context_tokens,
                'question': question,
                'question_tokens': question_tokens,
                'reference': reference,
                'reference_tokens': reference_tokens,
                'candidate': candidate,
                'candidate_tokens': candidate_tokens,
            })
        }
        if score:
            fields['score'] = ArrayField(np.array(score))

        return Instance(fields)

    def truncate_context(self, context, question, reference, candidate):
        """ Calculates if the current input would be over `self.max_length`
            and if so, truncates the context so that the input would be at
            `self.max_length`.
        """

        num_added_tokens = self.tokenizer.num_special_tokens_to_add(pair=True) + 2
        current_length = len(context) + len(question) + len(reference) + \
                         len(candidate) + num_added_tokens

        if current_length > self.max_length:
            difference = self.max_length - current_length
            context = context[:difference]

        return context

    def create_input(self, context, question, reference, candidate):
        # `input_tokens`: `[CLS] cont [SEP] ques [SEP]  ref [SEP] cand [SEP]`
        cls = [self.tokenizer.cls_token]
        sep = [self.tokenizer.sep_token]
        input_tokens = cls + context + sep + question + sep + reference + sep + candidate + sep
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # `token_type_ids`: is 0 for `[CLS] cont [SEP] ques [SEP]` and
        #                   1 for `ref [SEP] cand [SEP]`
        token_type_ids = [0] * (len(context) + len(question) + 3) +  \
                         [1] * (len(reference) + len(candidate) + 2)

        # `attention_mask` is 1's for all positions which aren't padding
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(token_type_ids) == len(attention_mask)

        return input_ids, token_type_ids, attention_mask
