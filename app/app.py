from flask import Flask

from .handlers import api

app = Flask(__name__)
app.config["RESTX_VALIDATE"] = True
app.config["RESTX_MASK_SWAGGER"] = False
api.init_app(app)
