import random
import uuid
from dataclasses import asdict
from typing import Generator, List
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from app.app import app
from app.dtos import ModelMetadata, PredictionResult, TrainResult
from app.services import ModelService
from app.services.model_service import score_funcs


class TestModels:
    @pytest.fixture
    def client(self) -> Generator[FlaskClient, None, None]:
        with app.test_client() as client:
            yield client

    @pytest.fixture
    def three_models(self) -> List[ModelMetadata]:
        models = []
        for i in range(3):
            model_class = random.choice(["logistic", "linear"])
            models.append(
                ModelMetadata(
                    model_class=model_class,
                    score_func=random.choice(score_funcs[model_class]),
                    num_features=10,
                    learning_rate=random.random(),
                    k=random.randint(1, 10) if i % 2 else None,
                )
            )
        return models

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
        model_id = str(uuid.uuid4())
        with patch.object(
            ModelService,
            "train",
            return_value=TrainResult(model_id=model_id, train_acc=0.5, valid_acc=0.5),
        ):
            resp = client.post(
                url,
                json={
                    "model_class": "logistic",
                    "score_func": "f_classif",
                    "num_features": 10,
                    "learning_rate": 0.5,
                    "k": 2,
                },
            )
            data = resp.get_json()
            assert resp.status_code == 201
            assert data["model_id"] == model_id
            assert 0 <= data["train_acc"] <= 1
            assert 0 <= data["valid_acc"] <= 1

        model_id = str(uuid.uuid4())
        with patch.object(
            ModelService,
            "train",
            return_value=TrainResult(model_id=model_id, train_acc=0.5, valid_acc=0.5),
        ):
            resp = client.post(
                url,
                json={
                    "model_class": "linear",
                    "score_func": "f_regression",
                    "num_features": 10,
                    "learning_rate": 2.5,
                    "k": 10,
                },
            )
            data = resp.get_json()
            assert resp.status_code == 201
            assert data["model_id"] == model_id
            assert 0 <= data["train_acc"] <= 1
            assert 0 <= data["valid_acc"] <= 1

        # Invalid ModelMetadata input
        resp = client.post(
            url,
            json={
                "model_class": "RandomForest",
                "score_func": "SelectFpr",
                "learning_rate": -0.1,
                "k": -1,
            },
        )
        assert resp.status_code == 400

    def test_get_model(self, client: FlaskClient) -> None:
        url = "/api/models/{}"

        # Model ID must be an UUID
        resp = client.get(url.format("abcd"))
        assert resp.status_code == 400

        # Model must exist
        model_id = str(uuid.uuid4())
        with patch.object(ModelService, "get_model", return_value=None):
            resp = client.get(url.format(model_id))
            assert resp.status_code == 404

        # Returns desired data
        with patch.object(
            ModelService,
            "get_model",
            return_value=ModelMetadata(
                model_class="linear",
                score_func="f_regression",
                num_features=10,
                learning_rate=0.5,
                k=2,
            ),
        ):
            resp = client.get(url.format(model_id))
            data = resp.get_json()
            assert resp.status_code == 200
            assert data["model_class"] == "linear"
            assert data["score_func"] == "f_regression"
            assert data["num_features"] == 10
            assert 0 <= data["learning_rate"] <= 1
            assert data["k"] == 2

    def test_delete_model(self, client: FlaskClient) -> None:
        url = "/api/models/{}"

        # Model ID must be an UUID
        resp = client.get(url.format("abcd"))
        assert resp.status_code == 400

        # Model must exist to be deleted
        model_id = str(uuid.uuid4())
        with patch.object(ModelService, "get_model", return_value=None):
            resp = client.delete(url.format(model_id))
            assert resp.status_code == 404

        # Successful deletion should return 204
        with patch.object(
            ModelService,
            "get_model",
            return_value=ModelMetadata(
                model_class="logistic",
                score_func="f_classif",
                num_features=10,
                learning_rate=0.5,
            ),
        ):
            resp = client.delete(url.format(model_id))
            assert resp.status_code == 204

    def test_predict(self, client: FlaskClient) -> None:
        url = "/api/models/{}/predict"

        # Model ID must be an UUID
        resp = client.post(url.format("abcd"))
        assert resp.status_code == 400

        # Age must be between 15 and 22
        resp = client.post(url.format(1), json={"age": 40, "family_size": "LE3"})
        assert resp.status_code == 400

        applicant = {
            "school": "GP",
            "sex": "M",
            "age": 20,
            "family_size": "LE3",
            "absences": 50,
        }

        # Model must exist
        model_id = str(uuid.uuid4())
        with patch.object(ModelService, "get_model", return_value=None):
            resp = client.post(url.format(model_id), json=applicant)
            assert resp.status_code == 404

        # Returns desired data
        with patch.object(
            ModelService,
            "predict",
            return_value=PredictionResult(model_id=model_id, success=True),
        ):
            resp = client.post(url.format(model_id), json=applicant)
            data = resp.get_json()
            assert resp.status_code == 200
            assert data["model_id"] == model_id
            assert data["success"]

        model_id = str(uuid.uuid4())
        with patch.object(
            ModelService,
            "predict",
            return_value=PredictionResult(model_id=model_id, success=False),
        ):
            resp = client.post(url.format(model_id), json=applicant)
            data = resp.get_json()
            assert resp.status_code == 200
            assert data["model_id"] == model_id
            assert not data["success"]
