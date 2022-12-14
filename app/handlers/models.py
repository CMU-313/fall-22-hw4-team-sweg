import uuid
from dataclasses import asdict, fields
from typing import List, Tuple

from flask_restx import Namespace, Resource, reqparse

from app.dtos import (Applicant, ApplicantFields, ModelMetadata,
                      ModelMetadataFields, PredictionResult,
                      PredictionResultFields, TrainResult, TrainResultFields)
from app.dtos.train import TrainMetadata, TrainMetadataFields
from app.services import ModelService

api = Namespace(
    name="models", description="API endpoints to manage machine learning models"
)
applicant_model = api.model(name="Applicant", model=asdict(ApplicantFields()))
train_metadata_model = api.model(
    name="TrainMetadata", model=asdict(TrainMetadataFields())
)
model_metadata_model = api.model(
    name="ModelMetadata", model=asdict(ModelMetadataFields())
)
train_result_model = api.model(name="TrainResult", model=asdict(TrainResultFields()))
prediction_result_model = api.model(
    name="PredictionResult", model=asdict(PredictionResultFields())
)


@api.route("")
class ModelList(Resource):
    @api.marshal_with(model_metadata_model, as_list=True, code=200)
    def get(self) -> Tuple[List[ModelMetadata], int]:
        """Gets a list of all the models"""
        return ModelService.get_model_list(), 200

    @api.expect(train_metadata_model)
    @api.marshal_with(train_result_model, code=201)
    @api.response(400, "Invalid input")
    def post(self) -> Tuple[TrainResult, int]:
        """Creates and trains a model with given model class and hyperparameters"""
        parser = reqparse.RequestParser()
        parser.add_argument("model_class", required=True, type=str)
        parser.add_argument("score_func", required=True, type=str)
        parser.add_argument("num_features", required=True, type=int)
        parser.add_argument("k", required=True, type=int)
        args = parser.parse_args()

        try:
            return (
                ModelService.train(
                    TrainMetadata(
                        model_class=args["model_class"],
                        score_func=args["score_func"],
                        num_features=args["num_features"],
                        k=args["k"],
                    )
                ),
                201,
            )
        except ValueError as e:
            api.abort(400, str(e))


@api.route("/<model_id>")
@api.param("model_id", description="The model ID")
class Model(Resource):
    @api.marshal_with(model_metadata_model, code=200)
    @api.response(400, "Invalid input")
    @api.response(404, "Model does not exist")
    def get(self, model_id: str) -> Tuple[ModelMetadata, int]:
        """Gets a model with a given ID"""
        try:
            uuid.UUID(model_id, version=4)
        except ValueError:
            api.abort(400, "Invalid model ID")
        if not ModelService.get_model(model_id):
            api.abort(404, "Model does not exist")
        return ModelService.get_model(model_id), 200

    @api.response(204, "Success")
    @api.response(400, "Invalid input")
    @api.response(404, "Model does not exist")
    def delete(self, model_id: str) -> Tuple[str, int]:
        """Deletes a model with a given ID"""
        try:
            uuid.UUID(model_id, version=4)
        except ValueError:
            api.abort(400, "Invalid model ID")
        if not ModelService.get_model(model_id):
            api.abort(404, "Model does not exist")
        ModelService.delete(model_id)
        return "", 204


@api.route("/<model_id>/predict")
@api.param("model_id", description="The model ID")
class ModelPrediction(Resource):
    @api.expect(applicant_model)
    @api.marshal_with(prediction_result_model, code=200)
    @api.response(400, "Invalid input")
    @api.response(404, "Model does not exist")
    def post(self, model_id: str) -> Tuple[PredictionResult, int]:
        """Predicts the success of an applicant using a given model"""
        try:
            uuid.UUID(model_id, version=4)
        except ValueError:
            api.abort(400, "Invalid model ID")
        model_metadata = ModelService.get_model(model_id)
        if not model_metadata:
            api.abort(404, "Model does not exist")

        parser = reqparse.RequestParser()
        for field in fields(Applicant):
            parser.add_argument(field.name, type=field.type, location="json")
        args = parser.parse_args()
        try:
            return (
                ModelService.predict(model_id, model_metadata, Applicant(**args)),
                200,
            )
        except ValueError as e:
            api.abort(400, str(e))
