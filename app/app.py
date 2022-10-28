from flask import Flask
from .handlers import api

app = Flask(__name__)
api.init_app(app)
