from flask_restx import Api

from .models import api as models

api = Api(prefix="/api", doc="/api/docs")
api.add_namespace(models, path="/models")
