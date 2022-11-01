from typing import Optional, List

from app.dtos import Applicant, ModelMetadata, TrainResult, PredictionResult


class ModelService:

    @staticmethod
    def get_model(model_id: int) -> Optional[ModelMetadata]:
        return ModelMetadata(model_class="logistic", learning_rate=0.1, k=2)

    @staticmethod
    def get_model_list() -> list[ModelMetadata]:
        # empty list so far, but should fetch the data from a
        # specific directory later, and parse out the data into
        # a list
        return []
        
    @staticmethod
    def train(model_metadata: ModelMetadata) -> TrainResult:
        # TODO (jaehoon): Implement this function
        return TrainResult(model_id=1, train_acc=0.5, valid_acc=0.5)

    @staticmethod
    def predict(model_id: int, applicant: Applicant) -> PredictionResult:
        # TODO (kyungmin): Implement this function
        return PredictionResult(model_id=model_id, success=False)
