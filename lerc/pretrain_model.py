import logging
from overrides import overrides
from typing import Dict

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy
from transformers import BertModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("pretrain-lerc")
class PretrainLERC(Model):
    @property
    def embedding_dim(self):
        return self.bert.embeddings.word_embeddings.embedding_dim

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset)
                for metric_name, metric in self.metrics.items()}

    def __init__(
        self,
        bert_model: str = 'bert-base-uncased',
        vocab=Vocabulary(),
        initializer=InitializerApplicator()
    ) -> None:
        super(PretrainLERC, self).__init__(vocab)
        self.bert = BertModel.from_pretrained(bert_model)
        self.label_layer = torch.nn.Linear(self.embedding_dim, 3)
        self.metrics = {'accuracy': CategoricalAccuracy()}
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        label: torch.Tensor = None,
        metadata: Dict = None
    ) -> Dict:
        # output.size() = [batch_size, seq_len, embedding_dim]
        output, _ = self.bert(input_ids=input_ids,
                              token_type_ids=token_type_ids,
                              attention_mask=attention_mask)

        # cls_output.size() = [batch_size, embedding_dim]
        cls_output = output[:, 0, :]

        # logits.size() = [batch_size, 3]
        logits = self.label_layer(cls_output)

        output_dict = {
            'logits': logits,
            'class_probabilties': torch.nn.functional.softmax(logits, dim=-1),
            'pred_label': torch.max(logits, dim=-1)[1],
            'metadata': metadata
        }

        if label is not None:
            label = label.long()
            self.metrics['accuracy'](logits, label)
            output_dict['loss'] = self.loss(logits, label)
            output_dict['label'] = label

        return output_dict