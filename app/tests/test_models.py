from typing import Generator
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from app.app import app
from app.dtos import TrainResult
from app.services import ModelService


class TestModels:

    @pytest.fixture
    def client(self) -> Generator[FlaskClient, None, None]:
        with app.test_client() as client:
            yield client
    
    def test_post_model(self, client: FlaskClient) -> None:
        url = "/api/models"

        # Returns desired data
        with patch.object(ModelService, "train", return_value=TrainResult(model_id=1, train_acc=0.5, valid_acc=0.5)):
            resp = client.post(url, json={
                "model_class" : "logistic",
                "learning_rate" : 0.5,
                "k" : 2
            })
            data = resp.get_json()
            assert resp.status_code == 201
            assert data["model_id"] == 1
            assert 0 <= data["train_acc"] <= 1
            assert 0 <= data["valid_acc"] <= 1
        
        with patch.object(ModelService, "train", return_value=TrainResult(model_id=2, train_acc=0.5, valid_acc=0.5)):
            resp = client.post(url, json={
                "model_class" : "linear",
                "learning_rate" : 2.5,
                "k" : 10
            })
            data = resp.get_json()
            assert resp.status_code == 201
            assert data["model_id"] == 2
            assert 0 <= data["train_acc"] <= 1
            assert 0 <= data["valid_acc"] <= 1

        # Invalid ModelMetadata input
        resp = client.post(url, json={
            "model_class" : "RandomForest",
            "learning_rate" : -0.1,
            "k" : -1,
        })
        assert resp.status_code == 400


    def test_predict(self, client: FlaskClient) -> None:
        url = "/api/models/{}/predict"

        # Model ID must be an integer
        resp = client.post(url.format("abcd"))
        assert resp.status_code == 404

        # Model ID must be positive
        resp = client.post(url.format(0))
        data = resp.get_json()
        assert resp.status_code == 400

        # Age must be between 15 and 22
        resp = client.post(url.format(1),
                           json={
                               "age": 40,
                               "family_size": True
                           })
        assert resp.status_code == 400

        applicant = {
            "school": True,
            "sex": False,
            "age": 20,
            "family_size": True,
            "absences": 50,
        }

        # Model must exist
        with patch.object(ModelService, "get_model", return_value=None):
            resp = client.post(url.format(1), json=applicant)
            data = resp.get_json()
            assert resp.status_code == 404

        # Returns desired data
        with patch.object(ModelService, "predict", return_value=True):
            resp = client.post(url.format(1), json=applicant)
            data = resp.get_json()
            assert resp.status_code == 200
            assert data["model_id"] == 1
            assert data["success"]

        with patch.object(ModelService, "predict", return_value=False):
            resp = client.post(url.format(2), json=applicant)
            data = resp.get_json()
            assert resp.status_code == 200
            assert data["model_id"] == 2
            assert not data["success"]
    


