"""Tests for the model artifact location (issue #28).

Training no longer happens at Docker build time; CI trains the model and
deployments mount the artifact. These tests pin down the contract that
makes that work: one resolved artifact path shared by training and
serving, overridable with MODEL_PATH.
"""

from model import DEFAULT_ARTIFACT_PATH, model_artifact_path


def test_default_artifact_lives_in_artifacts_subdir():
    # The artifact must not sit directly in model/ — deployments mount a
    # volume over the artifact directory, and mounting over model/ would
    # shadow the package's code.
    assert model_artifact_path() == DEFAULT_ARTIFACT_PATH
    assert DEFAULT_ARTIFACT_PATH.parent.name == "artifacts"
    assert DEFAULT_ARTIFACT_PATH.name == "model.json"


def test_model_path_env_overrides_default(monkeypatch, tmp_path):
    override = tmp_path / "elsewhere" / "model.json"
    monkeypatch.setenv("MODEL_PATH", str(override))
    assert model_artifact_path() == override


def test_load_model_returns_none_when_artifact_missing(monkeypatch, tmp_path):
    # Serving must degrade to the heuristic, not crash, when no artifact
    # has been mounted or trained yet.
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.json"))
    from model.predict import load_model

    assert load_model() is None


def test_train_saves_and_predict_loads_same_artifact(monkeypatch, tmp_path):
    # End-to-end: training writes the artifact, serving loads it from the
    # same resolved location. Uses a tiny model to stay fast.
    import pandas as pd
    from model.predict import load_model
    from model.train import build_model

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "artifacts" / "model.json"))

    X = pd.DataFrame({
        "win_pct_diff": [0.1, -0.2, 0.3, -0.1, 0.0, 0.2],
        "point_diff_diff": [1.0, -2.0, 3.0, -1.0, 0.0, 2.0],
    })
    y = pd.Series([1, 0, 1, 0, 1, 1])
    model = build_model()
    model.set_params(n_estimators=5)
    model.fit(X, y)

    path = model_artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))

    loaded = load_model()
    assert loaded is not None
    assert loaded.predict(X).shape == (len(X),)
