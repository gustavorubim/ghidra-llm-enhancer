from __future__ import annotations

import logging
from pathlib import Path

import pytest
import typer

from decomp_clarifier import cli as cli_module


def test_bootstrap_writes_app_config(tmp_path: Path, temp_app_config, monkeypatch) -> None:
    logger = logging.getLogger("bootstrap-test")
    logger.handlers.clear()

    monkeypatch.setattr(cli_module.ProjectPaths, "discover", lambda: tmp_path)
    monkeypatch.setattr(cli_module, "load_dotenv", lambda root: None)
    monkeypatch.setattr(cli_module, "load_app_config", lambda root, name="default": temp_app_config)
    monkeypatch.setattr(
        cli_module,
        "configure_logging",
        lambda level, log_file, log_to_console: logger,
    )

    root, paths, run_id, run_dir, configured_logger, app_config = cli_module._bootstrap("unit")

    assert root == tmp_path
    assert paths.root == tmp_path
    assert run_id.startswith("unit-")
    assert run_dir.exists()
    assert configured_logger is logger
    assert app_config == temp_app_config
    assert (run_dir / "app_config.yaml").exists()


def test_cli_loader_helpers_round_trip(
    temp_paths, sample_project, sample_compile_manifest, sample_dataset_samples
) -> None:
    project_dir = temp_paths.generated_projects_dir / sample_project.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_manifest.json").write_text(
        sample_project.model_dump_json(),
        encoding="utf-8",
    )

    binary_dir = temp_paths.binaries_dir / sample_project.project_id
    binary_dir.mkdir(parents=True, exist_ok=True)
    (binary_dir / "compile_manifest.json").write_text(
        sample_compile_manifest.model_dump_json(),
        encoding="utf-8",
    )

    dataset_path = temp_paths.processed_sft_dir / "function_dataset.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "\n".join(sample.model_dump_json() for sample in sample_dataset_samples[:2]) + "\n",
        encoding="utf-8",
    )

    assert cli_module._load_generated_projects(temp_paths)[0].project_id == sample_project.project_id
    assert (
        cli_module._load_compile_manifests(temp_paths)[0].project_id
        == sample_compile_manifest.project_id
    )
    assert len(cli_module._load_dataset_samples(dataset_path)) == 2


def test_quarantine_project_moves_artifacts(temp_paths, sample_project) -> None:
    project_dir = temp_paths.generated_projects_dir / sample_project.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "main.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")

    manifest_path = temp_paths.manifests_dir / f"{sample_project.project_id}.json"
    manifest_path.write_text('{"project_id":"sample_project"}', encoding="utf-8")

    binary_dir = temp_paths.binaries_dir / sample_project.project_id
    binary_dir.mkdir(parents=True, exist_ok=True)
    (binary_dir / "sample_project.exe").write_text("", encoding="utf-8")

    cli_module._quarantine_project(temp_paths, sample_project.project_id)

    assert not project_dir.exists()
    assert not manifest_path.exists()
    assert not binary_dir.exists()
    assert (
        temp_paths.generated_projects_dir / "_quarantine" / sample_project.project_id / "main.c"
    ).exists()
    assert (
        temp_paths.manifests_dir / "quarantine" / f"{sample_project.project_id}.json"
    ).exists()
    assert (temp_paths.binaries_dir / "_quarantine" / sample_project.project_id).exists()


def test_ensure_compiler_available_exits_on_missing_compiler() -> None:
    class BrokenCompiler:
        @property
        def executable(self) -> str:
            raise FileNotFoundError("missing compiler")

    with pytest.raises(typer.Exit):
        cli_module._ensure_compiler_available(BrokenCompiler())  # type: ignore[arg-type]
