from dataclasses import asdict
from typing import List, Tuple

from flask_restx import Namespace, Resource

from app.dtos import (Applicant, ApplicantFields, ModelMetadata,
                      ModelMetadataFields, PredictionResult,
                      PredictionResultFields, TrainResult, TrainResultFields)
from app.services import ModelService

api = Namespace(
    name="models", description="API endpoints to manage machine learning models"
)
applicant = api.model(name="Applicant", model=asdict(ApplicantFields()))
model_metadata = api.model(name="ModelMetadata", model=asdict(ModelMetadataFields()))
train_result = api.model(name="TrainResult", model=asdict(TrainResultFields()))
prediction_result = api.model(
    name="PredictionResult", model=asdict(PredictionResultFields())
)


@api.route("")
class ModelList(Resource):
    @api.marshal_with(model_metadata, as_list=True, code=200)
    def get(self) -> Tuple[List[ModelMetadata], int]:
        """Gets a list of all the models"""
        return ModelService.get_model_list(), 200

    @api.expect(model_metadata)
    @api.marshal_with(train_result, code=201)
    @api.response(400, "Invalid input")
    def post(self) -> Tuple[TrainResult, int]:
        """Creates and trains a model with given model class and hyperparameters"""
        return (
            ModelService.train(ModelMetadata(model_class="linear", learning_rate=0.5)),
            201,
        )


@api.route("/<int:model_id>")
@api.param("model_id", description="The model ID")
class Model(Resource):
    @api.marshal_with(model_metadata, code=200)
    @api.response(400, "Invalid input")
    @api.response(404, "Model does not exist")
    def get(self, model_id: int) -> Tuple[ModelMetadata, int]:
        """Gets a model with a given ID"""
        if model_id <= 0:
            api.abort(400, "Invalid model ID")
        if not ModelService.get_model(model_id):
            api.abort(404, "Model does not exist")
        return ModelService.get_model(model_id), 200

    @api.response(204, "Success")
    @api.response(400, "Invalid input")
    @api.response(404, "Model does not exist")
    def delete(self, model_id: int) -> Tuple[str, int]:
        """Deletes a model with a given ID"""
        if model_id <= 0:
            api.abort(400, "Invalid model ID")
        if not ModelService.get_model(model_id):
            api.abort(404, "Model does not exist")
        ModelService.delete(model_id)
        return "", 204


@api.route("/<int:model_id>/predict")
@api.param("model_id", description="The model ID")
class ModelPrediction(Resource):
    @api.expect(applicant)
    @api.marshal_with(prediction_result, code=200)
    @api.response(400, "Invalid input")
    @api.response(404, "Model does not exist")
    def post(self, model_id: int) -> Tuple[PredictionResult, int]:
        """Predicts the success of an applicant using a given model"""
        if model_id <= 0:
            api.abort(400, "Invalid model ID")
        # TODO (kyungmin): Implement the endpoint
        if not ModelService.get_model(model_id):
            api.abort(404, "Model does not exist")
        return ModelService.predict(model_id, Applicant()), 200
