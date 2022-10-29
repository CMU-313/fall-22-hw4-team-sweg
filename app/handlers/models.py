from dataclasses import asdict
from typing import Any, Dict

from flask_restx import Namespace, Resource

from app.dtos import (
    ApplicantFields,
    ModelMetadataFields,
    PredictionResultFields,
    TrainResultFields,
)
from app.services import ModelService

api = Namespace(name="models", description="Models API")
applicant = api.model(name="Applicant", model=asdict(ApplicantFields()))
model_metadata = api.model(name="ModelMetadata", model=asdict(ModelMetadataFields()))
train_result = api.model(name="TrainResult", model=asdict(TrainResultFields()))
prediction_result = api.model(
    name="PredictionResult", model=asdict(PredictionResultFields())
)


@api.route("")
class ModelList(Resource):
    def get(self) -> Dict[str, Any]:
        # TODO (jihyo): Function Comment
        return {}

    @api.expect(model_metadata)
    @api.marshal_with(train_result, code=201)
    @api.response(400, "Invalid input")
    def post(self) -> Dict[str, Any]:
        """Trains a model with the client specified model class and hyperparameters"""
        return {"model_id": 1, "train_acc": 0.1, "valid_acc": 0.1}


@api.route("/<int:model_id>/predict")
@api.param("model_id", description="The model ID")
class ModelPrediction(Resource):
    @api.expect(applicant)
    @api.marshal_with(prediction_result, code=200)
    @api.response(400, "Invalid input")
    @api.response(404, "Model does not exist")
    def post(self, model_id: int) -> Dict[str, Any]:
        """Predicts the success of an applicant using a given model"""
        if model_id <= 0:
            api.abort(400, "Invalid model ID")
        # TODO (kyungmin): Implement the endpoint
        if not ModelService.get_model(model_id):
            api.abort(404, "Model does not exist")
        return {"model_id": model_id, "success": ModelService.predict(model_id, {})}
