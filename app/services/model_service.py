from typing import Optional

from app.dtos import Applicant, ModelMetadata, TrainResult


class ModelService:

    @staticmethod
    def get_model(model_id: int) -> Optional[ModelMetadata]:
        # TODO (jihyo): Implement this function
        return ModelMetadata(model_class="logistic", learning_rate=0.1, k=2)

    def train(model_metadata: ModelMetadata) -> TrainResult:
        # TODO (jaehoon) : Implement this function
        return TrainResult(model_id=1, train_acc=0.5, valid_acc=0.5)

    @staticmethod
    def predict(model_id: int, applicant: Applicant) -> bool:
        # TODO (kyungmin): Implement this function
        return False

    @staticmethod
    def delete(model_id: int) -> None:
        # TODO (victor): Implement this function
        return None