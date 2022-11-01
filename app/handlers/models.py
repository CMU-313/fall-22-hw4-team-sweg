from dataclasses import asdict
from typing import Any, Dict, Tuple

from flask_restx import Namespace, Resource

from app.dtos import (
    ApplicantFields,
    ModelMetadata,
    ModelMetadataFields,
    PredictionResultFields,
    TrainResultFields,
)
from app.services import ModelService

api = Namespace(name="models", description="Models API")
applicant = api.model(name="Applicant", model=asdict(ApplicantFields()))
model_metadata = api.model(name="ModelMetadata",
                           model=asdict(ModelMetadataFields()))
train_result = api.model(name="TrainResult", model=asdict(TrainResultFields()))
prediction_result = api.model(name="PredictionResult",
                              model=asdict(PredictionResultFields()))


@api.route("")
class ModelList(Resource):
    @api.marshal_with(model_metadata, code=200)
    def get(self) -> Dict[str, Any]:
        """Gets the list of modelMetadata"""
        return ModelService.get_model_list()

    @api.expect(model_metadata)
    @api.marshal_with(train_result, code=201)
    @api.response(400, "Invalid input")
    def post(self) -> Tuple[Dict[str, Any], int]:
        """Trains a model with the client specified model class and hyperparameters"""
        return ModelService.train(
            ModelMetadata(model_class="linear", learning_rate=0.5)), 201


@api.route("/<int:model_id>/predict")
@api.param("model_id", description="The model ID")
class ModelPrediction(Resource):

    @api.expect(applicant)
    @api.marshal_with(prediction_result, code=200)
    @api.response(400, "Invalid input")
    @api.response(404, "Model does not exist")
    def post(self, model_id: int) -> Tuple[Dict[str, Any], int]:
        """Predicts the success of an applicant using a given model"""
        if model_id <= 0:
            api.abort(400, "Invalid model ID")
        # TODO (kyungmin): Implement the endpoint
        if not ModelService.get_model(model_id):
            api.abort(404, "Model does not exist")
        return {
            "model_id": model_id,
            "success": ModelService.predict(model_id, {}),
        }, 200
