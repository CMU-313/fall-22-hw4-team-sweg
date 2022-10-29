from typing import Generator
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from app.app import app
from app.services import ModelService


class TestModels:

    @pytest.fixture
    def client(self) -> Generator[FlaskClient, None, None]:
        with app.test_client() as client:
            yield client

    def test_predict(self, client: FlaskClient) -> None:
        url = "/api/models/{}/predict"

        # Model ID must be an integer
        resp = client.post(url.format("abcd"))
        assert resp.status_code == 404

        # Model ID must be positive
        resp = client.post(url.format(0))
        data = resp.get_json()
        assert resp.status_code == 400
        assert data["message"] == "Invalid model ID"

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
            assert data["message"] == "Model does not exist"

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
