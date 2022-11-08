from flask import Flask

from .handlers import api

app = Flask(__name__)
app.config["RESTX_VALIDATE"] = True
app.config["RESTX_MASK_SWAGGER"] = False
app.config["SWAGGER_UI_DOC_EXPANSION"] = "list"
api.init_app(app)
