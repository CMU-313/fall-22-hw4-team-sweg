from flask_restx import Api

from .models import api as models

api = Api(
    title="Team SWEg API",
    description="API endpoints for a microservice that predicts potential applicant success using machine learning.",
    contact="Team SWEg (Eric Fan, Jihyo Chung, Kyungmin Kim, Leo Jung, Victor Waddell)",
    contact_url="https://github.com/CMU-313/fall-22-hw4-team-sweg",
    prefix="/api",
    doc="/api/docs",
)
api.add_namespace(models, path="/models")
