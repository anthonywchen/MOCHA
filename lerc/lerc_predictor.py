from overrides import overrides

from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from lerc.lerc_model import LERC
from lerc.lerc_dataset_reader import LERCDatasetReader
from lerc.pretrain_model import PretrainLERC


@Predictor.register("lerc")
class LERCPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def _json_to_instance(self, inputs) -> Instance:
        inputs = {
            'context': inputs['context'],
            'question': inputs['question'],
            'reference': inputs['reference'],
            'candidate': inputs['candidate']
        }
        return self._dataset_reader.text_to_instance(**inputs)
