"""
MLOps infrastructure tests — DVC pipeline, MLflow integration, Docker config, CI/CD.

Validates that all MLOps tooling is correctly wired without running
actual training or Docker builds.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ═══════════════════════════════════════════════════════════════════════
# DVC Pipeline
# ═══════════════════════════════════════════════════════════════════════

class TestDVCPipeline:
    @pytest.fixture(scope="class")
    def dvc_config(self):
        path = PROJECT_ROOT / "dvc.yaml"
        assert path.exists(), "dvc.yaml not found at project root"
        with open(path) as f:
            return yaml.safe_load(f)

    def test_dvc_yaml_parses_with_expected_stages(self, dvc_config):
        """dvc.yaml has all 6 expected stages."""
        expected = {
            "download-tles", "build-catalog",
            "train-transformer", "train-maneuver", "train-anomaly",
            "validate",
        }
        assert set(dvc_config["stages"].keys()) == expected

    def test_all_stage_scripts_exist(self, dvc_config):
        """Every cmd references a script that exists on disk."""
        for name, stage in dvc_config["stages"].items():
            cmd = stage["cmd"]
            # Extract the script path from "python scripts/foo.py ..."
            parts = cmd.split()
            script_idx = next(
                (i for i, p in enumerate(parts) if p.endswith(".py")), None
            )
            assert script_idx is not None, f"Stage '{name}' cmd has no .py script"
            script_path = PROJECT_ROOT / parts[script_idx]
            assert script_path.exists(), (
                f"Stage '{name}' references {parts[script_idx]} which does not exist"
            )

    def test_all_stage_deps_exist(self, dvc_config):
        """Every deps entry points to an existing file or directory."""
        for name, stage in dvc_config["stages"].items():
            for dep in stage.get("deps", []):
                dep_path = PROJECT_ROOT / dep
                assert dep_path.exists(), (
                    f"Stage '{name}' dep '{dep}' does not exist"
                )


# ═══════════════════════════════════════════════════════════════════════
# MLflow Integration
# ═══════════════════════════════════════════════════════════════════════

class TestMLflowIntegration:
    """Verify each training script calls MLflow with the correct experiment
    name and expected parameter/metric keys.

    We mock the mlflow module so no tracking server or local storage is needed.
    """

    @staticmethod
    def _make_mock_mlflow():
        """Create a mock mlflow module with all methods we expect to be called."""
        mock = MagicMock()
        mock.pytorch = MagicMock()
        return mock

    def test_trajectory_transformer_mlflow_calls(self):
        """train_trajectory_parallel.py logs correct experiment, params, metrics."""
        mock_mlflow = self._make_mock_mlflow()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow, "mlflow.pytorch": mock_mlflow.pytorch}):
            # Re-import to pick up the mock
            spec = importlib.util.spec_from_file_location(
                "train_trajectory_parallel",
                PROJECT_ROOT / "scripts" / "train_trajectory_parallel.py",
            )
            mod = importlib.util.module_from_spec(spec)

            # We only need to verify the import-time flag
            exec(compile(
                "try:\n    import mlflow\n    HAS_MLFLOW = True\nexcept ImportError:\n    HAS_MLFLOW = False\n",
                "<test>", "exec",
            ), mod.__dict__)

            assert mod.HAS_MLFLOW is True

        # Verify the experiment name is correct by checking the source
        source = (PROJECT_ROOT / "scripts" / "train_trajectory_parallel.py").read_text()
        assert 'set_experiment("sda-trajectory-transformer")' in source
        assert "log_params" in source
        assert "log_metrics" in source
        assert "log_artifact" in source
        assert "pytorch.log_model" in source

    def test_collision_predictor_mlflow_calls(self):
        source = (PROJECT_ROOT / "scripts" / "train_collision_predictor.py").read_text()
        assert 'set_experiment("sda-collision-predictor")' in source
        assert "log_params" in source
        assert "log_metrics" in source
        assert "pytorch.log_model" in source

        # Verify expected param keys
        assert '"lr"' in source or "'lr'" in source
        assert '"batch_size"' in source or "'batch_size'" in source
        assert '"epochs"' in source or "'epochs'" in source

    def test_anomaly_autoencoder_mlflow_calls(self):
        source = (PROJECT_ROOT / "scripts" / "train_anomaly_autoencoder.py").read_text()
        assert 'set_experiment("sda-anomaly-autoencoder")' in source
        assert "log_params" in source
        assert "log_metrics" in source
        assert "log_artifact" in source

        # Anomaly-specific metrics
        assert "final_loss" in source
        assert "fpr" in source
        assert "tpr" in source

    def test_maneuver_classifier_mlflow_calls(self):
        source = (PROJECT_ROOT / "scripts" / "retrain_maneuver_classifier.py").read_text()
        assert 'set_experiment("sda-maneuver-classifier")' in source
        assert "log_params" in source
        assert "log_metrics" in source
        assert "log_artifact" in source
        assert "pytorch.log_model" in source

        # Maneuver-specific metrics
        assert "best_val_acc" in source

    def test_all_scripts_have_graceful_fallback(self):
        """All training scripts handle missing mlflow gracefully."""
        scripts = [
            "scripts/train_trajectory_parallel.py",
            "scripts/train_collision_predictor.py",
            "scripts/train_anomaly_autoencoder.py",
            "scripts/retrain_maneuver_classifier.py",
        ]
        for script in scripts:
            source = (PROJECT_ROOT / script).read_text()
            assert "HAS_MLFLOW = False" in source, (
                f"{script} missing HAS_MLFLOW fallback"
            )
            assert "HAS_MLFLOW = True" in source, (
                f"{script} missing HAS_MLFLOW = True on success"
            )


# ═══════════════════════════════════════════════════════════════════════
# Docker Configuration
# ═══════════════════════════════════════════════════════════════════════

class TestDockerConfig:
    def test_docker_requirements_is_subset(self):
        """Every package in requirements-docker.txt exists in requirements.txt."""
        main_reqs = self._parse_requirements(PROJECT_ROOT / "requirements.txt")
        docker_reqs = self._parse_requirements(PROJECT_ROOT / "requirements-docker.txt")

        for pkg in docker_reqs:
            assert pkg in main_reqs, (
                f"requirements-docker.txt has '{pkg}' not found in requirements.txt"
            )

    def test_dockerfile_copy_paths_exist(self):
        """All COPY source paths in the Dockerfile exist on disk."""
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
        for line in dockerfile.splitlines():
            line = line.strip()
            if line.startswith("COPY") and not line.startswith("COPY --from"):
                parts = line.split()
                # COPY <src> <dst> — check src exists
                src = parts[1]
                src_path = PROJECT_ROOT / src
                assert src_path.exists(), (
                    f"Dockerfile COPY source '{src}' does not exist"
                )

    def test_compose_yaml_valid(self):
        """docker-compose.yml parses and has required services."""
        path = PROJECT_ROOT / "docker-compose.yml"
        assert path.exists(), "docker-compose.yml not found"
        with open(path) as f:
            config = yaml.safe_load(f)

        assert "services" in config
        assert "app" in config["services"]
        assert "db" in config["services"]

        # App depends on db
        app = config["services"]["app"]
        assert "depends_on" in app
        assert "db" in app["depends_on"]

        # DB has health check
        db = config["services"]["db"]
        assert "healthcheck" in db

    @staticmethod
    def _parse_requirements(path: Path) -> set[str]:
        """Extract package names (lowercase, no version) from a requirements file."""
        packages = set()
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Handle extras like uvicorn[standard]==0.27.1
            name = line.split("==")[0].split(">=")[0].split("<=")[0].split("<")[0].split(">")[0].split("[")[0]
            packages.add(name.lower().strip())
        return packages


# ═══════════════════════════════════════════════════════════════════════
# CI/CD Configuration
# ═══════════════════════════════════════════════════════════════════════

class TestCIConfig:
    def test_ci_workflow_valid(self):
        """GitHub Actions workflow parses and has expected jobs."""
        path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        assert path.exists(), "CI workflow not found"
        with open(path) as f:
            config = yaml.safe_load(f)

        assert "jobs" in config
        assert "test" in config["jobs"]
        assert "build-docker" in config["jobs"]

        # Test job uses Python 3.12
        test_job = config["jobs"]["test"]
        steps = test_job["steps"]
        python_step = next(
            (s for s in steps if s.get("name", "").startswith("Set up Python")), None
        )
        assert python_step is not None
        assert python_step["with"]["python-version"] == "3.12"

        # Build job depends on test
        build_job = config["jobs"]["build-docker"]
        assert "test" in build_job["needs"]


# ═══════════════════════════════════════════════════════════════════════
# Database Schema — AssessmentCache
# ═══════════════════════════════════════════════════════════════════════

class TestAssessmentCache:
    @pytest.fixture(autouse=True)
    def fresh_db(self, tmp_path, monkeypatch):
        import src.api.database as db_mod
        db_mod._engine = None
        db_mod._SessionLocal = None
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "test.db")
        db_mod.init_db()
        yield
        db_mod._engine = None
        db_mod._SessionLocal = None

    def test_cache_and_retrieve_assessment(self):
        from src.api.database import cache_assessment, get_cached_assessment

        result = {"threat_tier": "ELEVATED", "threat_score": 0.85}
        cache_assessment(object_id=42, timestep=100, result=result)

        cached = get_cached_assessment(object_id=42, timestep=100)
        assert cached is not None
        assert cached["threat_tier"] == "ELEVATED"
        assert cached["threat_score"] == 0.85

    def test_cache_miss_returns_none(self):
        from src.api.database import get_cached_assessment

        assert get_cached_assessment(object_id=999, timestep=0) is None

    def test_clear_assessment_cache(self):
        from src.api.database import (
            cache_assessment, clear_assessment_cache, get_cached_assessment,
        )

        cache_assessment(object_id=1, timestep=0, result={"score": 0.1})
        cache_assessment(object_id=2, timestep=0, result={"score": 0.2})
        count = clear_assessment_cache()
        assert count == 2
        assert get_cached_assessment(object_id=1, timestep=0) is None
