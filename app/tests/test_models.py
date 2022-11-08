import random
import uuid
from dataclasses import asdict
from typing import Any, Dict, Generator, List
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
    def applicant(self) -> Dict[str, Any]:
        return {
            "school": "GP",
            "sex": "F",
            "age": 22,
            "address": "U",
            "family_size": "LE3",
            "p_status": "T",
            "mother_edu": 4,
            "father_edu": 4,
            "mother_job": "teacher",
            "father_job": "teacher",
            "reason": "home",
            "guardian": "mother",
            "travel_time": 4,
            "study_time": 4,
            "failures": 4,
            "school_support": "yes",
            "family_support": "yes",
            "paid": "yes",
            "activities": "yes",
            "nursery": "yes",
            "higher": "yes",
            "internet": "yes",
            "romantic": "yes",
            "family_rel": 5,
            "free_time": 5,
            "going_out": 5,
            "workday_alcohol": 5,
            "weekend_alcohol": 5,
            "health": 5,
            "absences": 93,
        }

    @pytest.fixture
    def three_models(self) -> List[ModelMetadata]:
        models = []
        for i in range(3):
            model_class = random.choice(["logistic", "linear"])
            models.append(
                ModelMetadata(
                    model_id=str(uuid.uuid4()),
                    model_class=model_class,
                    score_func=random.choice(score_funcs[model_class]),
                    num_features=10,
                    k=random.randint(2, 10),
                    train_acc=random.random(),
                    valid_acc=random.random(),
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
                model_id=model_id,
                model_class="linear",
                score_func="f_regression",
                num_features=10,
                k=2,
                train_acc=0.5,
                valid_acc=0.5,
            ),
        ):
            resp = client.get(url.format(model_id))
            data = resp.get_json()
            assert resp.status_code == 200
            assert data["model_class"] == "linear"
            assert data["score_func"] == "f_regression"
            assert data["num_features"] == 10
            assert data["k"] == 2
            assert 0 <= data["train_acc"] <= 1
            assert 0 <= data["valid_acc"] <= 1

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
                model_id=model_id,
                model_class="logistic",
                score_func="f_classif",
                num_features=10,
                k=2,
                train_acc=0.5,
                valid_acc=0.5,
            ),
        ):
            resp = client.delete(url.format(model_id))
            assert resp.status_code == 204

    def test_predict(self, client: FlaskClient, applicant) -> None:
        url = "/api/models/{}/predict"

        # Model ID must be an UUID
        resp = client.post(url.format("abcd"))
        assert resp.status_code == 400

        # Age must be between 15 and 22
        applicant["age"] = 40
        resp = client.post(url.format(1), json=applicant)
        assert resp.status_code == 400
        applicant["age"] = 22

        # Model must exist
        model_id = str(uuid.uuid4())
        with patch.object(ModelService, "get_model", return_value=None):
            resp = client.post(url.format(model_id), json=applicant)
            assert resp.status_code == 404

        # Returns desired data
        model_metadata = ModelMetadata(
            model_id=model_id,
            train_acc=0.5,
            valid_acc=0.5,
            model_class="logistic",
            score_func="f_classif",
            num_features=25,
            k=5,
        )
        with patch.object(ModelService, "get_model", return_value=model_metadata):
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

        with patch.object(ModelService, "get_model", return_value=model_metadata):
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
