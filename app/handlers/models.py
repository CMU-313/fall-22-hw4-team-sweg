from flask_restx import Namespace, Resource

api = Namespace("models")


@api.route("/<id>")
@api.param("id", "The model identifier")
class Model(Resource):
    @api.doc("predict")
    def predict(self, id) -> None:
        pass
