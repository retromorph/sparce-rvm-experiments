import json
from pathlib import Path

from sparce.cli import main
from sparce.smoke import run_smoke


def test_smoke_is_synthetic_and_reversible():
    result = run_smoke()
    assert result["synthetic"] is True
    assert result["scope_mismatch_blocked"]
    assert result["snapshot_rollback_equal"]
    assert result["active_memory_count"] > 0


def test_cli_smoke(capsys):
    assert main(["smoke", "--no-mlflow"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["synthetic"] is True


def test_suite_dry_run(capsys):
    root = Path(__file__).resolve().parents[1]
    assert main(["run-suite", "--dry-run", "--config", str(root / "configs/a4000.toml")]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "SCAFFOLD_ONLY"
    assert data["experiments"] == ["E1", "E2", "E3"]


def test_unimplemented_suite_never_claims_real_results(capsys):
    root = Path(__file__).resolve().parents[1]
    assert main(["run-suite", "--config", str(root / "configs/a4000.toml")]) == 2
    assert "NOT IMPLEMENTED" in capsys.readouterr().err
