"""Model package: training, prediction, and artifact location."""

import os
from pathlib import Path

MODEL_DIR = Path(__file__).parent

# Default artifact location. Lives in its own subdirectory so deployments
# can mount a volume over just the artifact without shadowing this
# package's code (mounting over /app/model hid code updates behind
# stale volumes).
DEFAULT_ARTIFACT_PATH = MODEL_DIR / "artifacts" / "model.json"


def model_artifact_path() -> Path:
    """Resolved location of the trained model artifact.

    The MODEL_PATH environment variable overrides the default so CI,
    Docker, and local runs can agree on one artifact location.
    """
    return Path(os.environ.get("MODEL_PATH", DEFAULT_ARTIFACT_PATH))
