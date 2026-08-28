from __future__ import annotations

import json

from bnpl_credit_risk.models.registry import resolve_model_dir


def test_latest_model_is_resolved_relative_to_current_registry(tmp_path):
    models_dir = tmp_path / "models"
    version_dir = models_dir / "v1"
    version_dir.mkdir(parents=True)
    (models_dir / "latest.json").write_text(
        json.dumps({"version": "v1", "path": "/stale/absolute/path/v1"})
    )

    assert resolve_model_dir(models_dir, "latest") == version_dir
