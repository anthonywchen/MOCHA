import logging
import numpy as np
import random
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.instance import Instance
from jsonlines import Reader
from transformers import BertTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("pretrain-lerc")
class PretrainLERCDatasetReader(DatasetReader):
    def __init__(
        self,
        bert_model: str = 'bert-base-uncased',
        lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

    @overrides
    def _read(self, file_path: str):
        # Load lines and shuffle.
        lines = [l for l in Reader(open(file_path))]
        random.shuffle(lines)
        for line in lines:
            yield self.text_to_instance(**line)

    @overrides
    def text_to_instance(
        self, context, question, reference, candidate, same, dataset=None
    ) -> Instance:
        context_tokens = self.tokenizer.tokenize(context)
        question_tokens = self.tokenizer.tokenize(question)
        reference_tokens = self.tokenizer.tokenize(reference)
        candidate_tokens = self.tokenizer.tokenize(candidate)

        context_tokens = self.truncate_context(
            context_tokens, question_tokens, reference_tokens, candidate_tokens
        )

        # Correct answer is the first one
        if random.random() > 0.5:
            input_ids, token_type_ids, attention_mask = self.create_input(
                context_tokens, question_tokens, reference_tokens, candidate_tokens
            )
            label = 0
        # Correct answer is the second one
        else:
            input_ids, token_type_ids, attention_mask = self.create_input(
                context_tokens, question_tokens, candidate_tokens, reference_tokens
            )
            label = 1

        label = 2 if same else label

        fields = {
            'input_ids': ArrayField(np.array(input_ids), dtype=np.int64,
                                    padding_value=self.tokenizer.pad_token_id),
            'token_type_ids': ArrayField(np.array(token_type_ids),
                                         dtype=np.int64),
            'attention_mask': ArrayField(np.array(attention_mask),
                                         dtype=np.int64),
            'label': ArrayField(np.array(label)),
            'metadata': MetadataField({
                'context_tokens': context_tokens,
                'question_tokens': question_tokens,
                'reference_tokens': reference_tokens,
                'candidate_tokens': candidate_tokens,
                'dataset': dataset,
                'same': same,
            })
        }

        return Instance(fields)

    def truncate_context(self, context, question, reference, candidate):
        num_added_tokens = self.tokenizer.num_special_tokens_to_add(pair=True) + 2
        current_length = len(context) + len(question) + len(reference) + len(candidate) + num_added_tokens

        if current_length > self.tokenizer.max_len:
            difference = self.tokenizer.max_len - current_length
            context = context[:difference]

        return context

    def create_input(self, context, question, ans1, ans2):
        # `input_tokens`: `[CLS] cont [SEP] ques [SEP] ans1 [SEP] ans2 [SEP]`
        cls = [self.tokenizer.cls_token]
        sep = [self.tokenizer.sep_token]
        input_tokens = cls + context + sep + question + sep + ans1 + sep + ans2 + sep
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # `token_type_ids`: is 0 for `[CLS] cont [SEP] ques [SEP]` and
        #                   1 for `ans1 [SEP] ans2 [SEP]`
        token_type_ids = [0] * (len(context) + len(question) + 3) +  \
                         [1] * (len(ans1) + len(ans2) + 2)

        # `attention_mask` is 1's for all positions which aren't padding
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(token_type_ids) == len(attention_mask)

        return input_ids, token_type_ids, attention_mask