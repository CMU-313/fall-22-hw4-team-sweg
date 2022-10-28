from flask_restx import Api

from .models import api as models

api = Api()
api.add_namespace(models, path="/models")
