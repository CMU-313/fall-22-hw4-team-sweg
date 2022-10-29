from typing import Optional

from app.dtos import Applicant, ModelMetadata


class ModelService:
    @staticmethod
    def get_model(model_id: int) -> Optional[ModelMetadata]:
        # TODO (jihyo): Implement this function
        return ModelMetadata(model_class="logistic", learning_rate=0.1, k=2)

    @staticmethod
    def predict(model_id: int, applicant: Applicant) -> bool:
        # TODO (kyungmin): Implement this function
        return False
