"""Integration tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app, _env
from app.datasets import DatasetRegistry
from app.environment import DataCleaningEnv


@pytest.fixture
def client():
    """Create a test client with proper initialization."""
    # Initialize the environment directly for testing
    import app.main as main_module
    registry = DatasetRegistry()
    main_module._env = DataCleaningEnv(registry=registry)
    yield TestClient(app)
    main_module._env = None


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestResetEndpoint:
    def test_reset_easy(self, client):
        response = client.post("/reset", json={"difficulty": "easy"})
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert len(data["state"]) == 50  # Easy has 50 rows
        assert data["done"] is False
        assert data["info"]["difficulty"] == "easy"
        assert data["info"]["max_steps"] == 15

    def test_reset_medium(self, client):
        response = client.post("/reset", json={"difficulty": "medium"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["state"]) == 200
        assert data["info"]["max_steps"] == 30

    def test_reset_hard(self, client):
        response = client.post("/reset", json={"difficulty": "hard"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["state"]) == 500
        assert data["info"]["max_steps"] == 60

    def test_reset_invalid_difficulty(self, client):
        response = client.post("/reset", json={"difficulty": "impossible"})
        assert response.status_code == 422  # Validation error


class TestStepEndpoint:
    def test_step_drop_duplicates(self, client):
        # Reset first
        client.post("/reset", json={"difficulty": "easy"})

        # Take a step
        response = client.post("/step", json={
            "action_type": "drop_duplicates",
            "params": {}
        })
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert "reward" in data
        assert data["info"]["step"] == 1

    def test_step_with_params(self, client):
        client.post("/reset", json={"difficulty": "easy"})

        response = client.post("/step", json={
            "action_type": "strip_whitespace",
            "params": {"column": "name"}
        })
        assert response.status_code == 200

    def test_step_invalid_action(self, client):
        client.post("/reset", json={"difficulty": "easy"})

        response = client.post("/step", json={
            "action_type": "invalid_action",
            "params": {}
        })
        assert response.status_code == 422  # Validation error

    def test_step_invalid_column(self, client):
        client.post("/reset", json={"difficulty": "easy"})

        response = client.post("/step", json={
            "action_type": "strip_whitespace",
            "params": {"column": "nonexistent_column"}
        })
        assert response.status_code == 400

    def test_step_without_reset(self, client):
        # Reset first, then run to completion, then try again
        client.post("/reset", json={"difficulty": "easy"})
        # Run all steps to finish the episode
        for _ in range(15):
            resp = client.post("/step", json={
                "action_type": "drop_duplicates",
                "params": {}
            })
            if resp.json().get("done"):
                break
        
        # Now try to step after done - should fail
        response = client.post("/step", json={
            "action_type": "drop_duplicates",
            "params": {}
        })
        # Should fail because episode is done
        assert response.status_code == 400


class TestStateEndpoint:
    def test_state_after_reset(self, client):
        client.post("/reset", json={"difficulty": "easy"})

        response = client.get("/state")
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert len(data["state"]) == 50

    def test_state_after_step(self, client):
        client.post("/reset", json={"difficulty": "easy"})
        client.post("/step", json={
            "action_type": "drop_duplicates",
            "params": {}
        })

        response = client.get("/state")
        assert response.status_code == 200
        data = response.json()
        # Should have fewer rows after dropping duplicates
        assert len(data["state"]) < 50


class TestEndToEndEpisode:
    def test_complete_easy_episode(self, client):
        # Reset
        response = client.post("/reset", json={"difficulty": "easy"})
        assert response.status_code == 200
        initial_rows = len(response.json()["state"])

        # Run through all steps
        done = False
        steps = 0
        max_steps = 15

        while not done and steps < max_steps:
            response = client.post("/step", json={
                "action_type": "drop_duplicates",
                "params": {}
            })
            assert response.status_code == 200
            data = response.json()
            done = data["done"]
            steps += 1

        # Should complete
        assert done is True
        assert data["info"]["final_score"] is not None
        assert 0.0 <= data["info"]["final_score"] <= 1.0
