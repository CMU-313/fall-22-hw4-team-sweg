from flask_restx import Api

from .models import api as models

api = Api(
    title="Team SWEg API",
    description="API endpoints for a microservice that predicts potential applicant success using machine learning.",
    prefix="/api",
    doc="/api/docs",
)
api.add_namespace(models, path="/models")
