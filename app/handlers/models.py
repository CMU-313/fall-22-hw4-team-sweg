from dataclasses import asdict
from typing import Any, Dict

from flask_restx import Namespace, Resource

from app.dtos import Applicant, PredictionResult

api = Namespace("models", description="Models API")
applicant = api.model(name="Applicant", model=asdict(Applicant()))
prediction_result = api.model(name="PredictionResult", model=asdict(PredictionResult()))


@api.route("/<int:model_id>/predict")
@api.param("model_id", description="The model ID")
class ModelPrediction(Resource):
    @api.expect(applicant)
    @api.marshal_with(prediction_result, code=200)
    @api.response(400, "Invalid input")
    @api.response(404, "Model does not exist")
    def post(self, model_id: int) -> Dict[str, Any]:
        """Predicts the success of a student using a given model"""
        if model_id <= 0:
            api.abort(400, "Invalid model ID")
        return {"model_id": model_id, "success": False}
