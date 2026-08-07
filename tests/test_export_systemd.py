"""Test systemd export correctly outputs YAML."""

import yaml
from click.testing import CliRunner
from sparkrun.cli import main


def test_export_systemd_preserves_single_quotes(monkeypatch, tmp_path):
    """Test systemd export does not escape single quotes in YAML."""
    runner = CliRunner()

    # Mock detect_remote_sparkrun to avoid SSH
    monkeypatch.setattr(
        "sparkrun.cli._export._detect_remote_sparkrun",
        lambda host, ssh_kwargs, dry_run=False: ("/usr/local/bin/sparkrun", "/home/user"),
    )

    # A recipe whose env values are quoted strings. This used to name
    # `@official/qwen3-coder-next-int4-autoround-vllm`, which made the test
    # depend on git-cloning the official registry *and* on that remote recipe
    # keeping this exact env var. The subject here is the export's YAML
    # quoting, so a local file (find_recipe resolves a direct path) tests it
    # without either dependency.
    recipe = tmp_path / "quoted-env-vllm.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "name": "quoted-env-vllm",
                "model": "Qwen/Qwen3-1.7B",
                "runtime": "vllm",
                "container": "vllm/vllm-openai:latest",
                "env": {"VLLM_MARLIN_USE_ATOMIC_ADD": "1"},
            }
        )
    )

    result = runner.invoke(main, ["export", "systemd", str(recipe), "--hosts", "127.0.0.1"])

    assert result.exit_code == 0, result.output
    # Look for the env var VLLM_MARLIN_USE_ATOMIC_ADD which should have single quotes
    # Before the fix, it would be VLLM_MARLIN_USE_ATOMIC_ADD: '\''1'\''
    # After the fix, it should be VLLM_MARLIN_USE_ATOMIC_ADD: '1'
    assert "VLLM_MARLIN_USE_ATOMIC_ADD: '1'" in result.output
    assert "VLLM_MARLIN_USE_ATOMIC_ADD: '\\''1'\\''" not in result.output

    # Another test: verify the bash script structure uses << 'SPARKRUN_RECIPE_EOF'
    assert "<< 'SPARKRUN_RECIPE_EOF'" in result.output
