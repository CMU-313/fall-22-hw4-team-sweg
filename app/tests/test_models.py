import random
from dataclasses import asdict
from typing import Generator, List
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from app.app import app
from app.dtos import ModelMetadata, PredictionResult, TrainResult
from app.services import ModelService


class TestModels:
    @pytest.fixture
    def client(self) -> Generator[FlaskClient, None, None]:
        with app.test_client() as client:
            yield client

    @pytest.fixture
    def three_models(self) -> List[ModelMetadata]:
        return [
            ModelMetadata(
                model_class=random.choice(["logistic", "linear"]),
                num_features=10,
                learning_rate=random.random(),
                k=random.randint(1, 10) if i % 2 else None,
            )
            for i in range(3)
        ]

    def test_get_model_list(self, client: FlaskClient, three_models) -> None:
        url = "/api/models"

        # Empty list
        with patch.object(ModelService, "get_model_list", return_value=[]):
            resp = client.get(url)
            data = resp.get_json()
            assert resp.status_code == 200
            assert data == []

        # Non-empty list
        with patch.object(ModelService, "get_model_list", return_value=three_models):
            resp = client.get(url)
            data = resp.get_json()
            assert resp.status_code == 200
            assert type(data) == list
            assert len(data) == len(three_models)
            assert all(m1 == asdict(m2) for m1, m2 in zip(data, three_models))

    def test_create_model(self, client: FlaskClient) -> None:
        url = "/api/models"

        # Returns desired data
        with patch.object(
            ModelService,
            "train",
            return_value=TrainResult(model_id=1, train_acc=0.5, valid_acc=0.5),
        ):
            resp = client.post(
                url, json={"model_class": "logistic", "num_features": 10, "learning_rate": 0.5, "k": 2}
            )
            data = resp.get_json()
            assert resp.status_code == 201
            assert data["model_id"] == 1
            assert 0 <= data["train_acc"] <= 1
            assert 0 <= data["valid_acc"] <= 1

        with patch.object(
            ModelService,
            "train",
            return_value=TrainResult(model_id=2, train_acc=0.5, valid_acc=0.5),
        ):
            resp = client.post(
                url, json={"model_class": "linear", "num_features": 10, "learning_rate": 2.5, "k": 10}
            )
            data = resp.get_json()
            assert resp.status_code == 201
            assert data["model_id"] == 2
            assert 0 <= data["train_acc"] <= 1
            assert 0 <= data["valid_acc"] <= 1

        # Invalid ModelMetadata input
        resp = client.post(
            url,
            json={
                "model_class": "RandomForest",
                "learning_rate": -0.1,
                "k": -1,
            },
        )
        assert resp.status_code == 400

    def test_get_model(self, client: FlaskClient) -> None:
        url = "/api/models/{}"

        # Model ID must be a number
        resp = client.get(url.format("abcd"))
        assert resp.status_code == 404

        # Model ID must be positive
        resp = client.get(url.format(0))
        assert resp.status_code == 400

        # Model must exist
        with patch.object(ModelService, "get_model", return_value=None):
            resp = client.get(url.format(1))
            assert resp.status_code == 404

        # Returns desired data
        with patch.object(
            ModelService,
            "get_model",
            return_value=ModelMetadata(model_class="linear", num_features=10, learning_rate=0.5, k=2),
        ):
            resp = client.get(url.format(1))
            data = resp.get_json()
            assert resp.status_code == 200
            assert data["model_class"] == "linear"
            assert data["num_features"] == 10
            assert 0 <= data["learning_rate"] <= 1
            assert data["k"] == 2

    def test_delete_model(self, client: FlaskClient) -> None:
        url = "/api/models/{}"

        # Model ID must be an integer
        resp = client.delete(url.format("testinginput"))
        assert resp.status_code == 404

        # Model ID must be positive
        resp = client.delete(url.format(0))
        assert resp.status_code == 400

        # Model must exist to be deleted
        with patch.object(ModelService, "get_model", return_value=None):
            resp = client.delete(url.format(1))
            assert resp.status_code == 404

        # Successful deletion should return 204
        with patch.object(
            ModelService,
            "get_model",
            return_value=ModelMetadata(model_class="logistic", num_features=10, learning_rate=0.5),
        ):
            resp = client.delete(url.format(1))
            assert resp.status_code == 204

    def test_predict(self, client: FlaskClient) -> None:
        url = "/api/models/{}/predict"

        # Model ID must be an integer
        resp = client.post(url.format("abcd"))
        assert resp.status_code == 404

        # Model ID must be positive
        resp = client.post(url.format(0))
        assert resp.status_code == 400

        # Age must be between 15 and 22
        resp = client.post(url.format(1), json={"age": 40, "family_size": True})
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
            assert resp.status_code == 404

        # Returns desired data
        with patch.object(
            ModelService,
            "predict",
            return_value=PredictionResult(model_id=1, success=True),
        ):
            resp = client.post(url.format(1), json=applicant)
            data = resp.get_json()
            assert resp.status_code == 200
            assert data["model_id"] == 1
            assert data["success"]

        with patch.object(
            ModelService,
            "predict",
            return_value=PredictionResult(model_id=2, success=False),
        ):
            resp = client.post(url.format(2), json=applicant)
            data = resp.get_json()
            assert resp.status_code == 200
            assert data["model_id"] == 2
            assert not data["success"]
