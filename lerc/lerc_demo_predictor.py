from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

from lerc.lerc_model import LERC
from lerc.lerc_dataset_reader import LERCDatasetReader
from lerc.pretrain_model import PretrainLERC


@Predictor.register("lerc_demo")
class LERCDemoPredictor(Predictor):
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

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """ Normalize prediction score between 0 and 1 """
        instance = self._json_to_instance(inputs)
        outputs = self.predict_instance(instance)
        outputs['pred_score'] = self.normalize(outputs['pred_score'])
        return outputs

    def normalize(self, pred_score):
        pred_score = (pred_score - 1)/4
        pred_score = min(1, max(0, pred_score))
        return pred_score
