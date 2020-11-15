import logging
from overrides import overrides
from transformers import BertModel
import torch
from typing import Dict

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics.pearson_correlation import PearsonCorrelation

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("lerc")
class LERC(Model):
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
        pretrained_archive_path: str = None,
        vocab=Vocabulary(),
        initializer=InitializerApplicator()
    ) -> None:
        super(LERC, self).__init__(vocab)
        if pretrained_archive_path:
            logger.info('Loading pretrained: %s', pretrained_archive_path)
            archive = load_archive(pretrained_archive_path)
            self.bert = archive.model.bert
        else:
            self.bert = BertModel.from_pretrained(bert_model)

        self.score_layer = torch.nn.Linear(self.embedding_dim, 1)
        self.metrics = {'pearson': PearsonCorrelation()}
        self.loss = torch.nn.MSELoss()
        initializer(self)

    @overrides
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        score: torch.Tensor = None,
        metadata: Dict = None
    ) -> Dict:
        output, _ = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        cls_output = output[:, 0, :]
        pred_score = self.score_layer(cls_output).squeeze(-1)

        output_dict = {'pred_score': pred_score, 'metadata': metadata}

        if score is not None:
            score = score.float()
            self.metrics['pearson'](pred_score, score)
            output_dict['loss'] = self.loss(pred_score, score)
            output_dict['score'] = score

        return output_dict