from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from decomp_clarifier.generation.canonicalize import canonicalize_project
from decomp_clarifier.generation.prompt_builder import (
    build_project_generation_prompt,
    build_project_repair_prompt,
)
from decomp_clarifier.generation.validators import validate_project
from decomp_clarifier.schemas.compiler import CompileManifest
from decomp_clarifier.schemas.generation import GeneratedProject
from decomp_clarifier.settings import GenerationConfig


class ProjectGenerator:
    def __init__(
        self,
        client: Any,
        config: GenerationConfig,
        prompt_template: str,
        repair_prompt_template: str | None,
        project_root: Path,
        manifest_root: Path,
    ) -> None:
        self.client = client
        self.config = config
        self.prompt_template = prompt_template
        self.repair_prompt_template = repair_prompt_template
        self.project_root = project_root
        self.manifest_root = manifest_root
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.manifest_root.mkdir(parents=True, exist_ok=True)

    def generate_many(self, count: int | None = None) -> list[GeneratedProject]:
        return [
            self.generate_one(index=index)
            for index in range(count or self.config.generation.project_count)
        ]

    def generate_one(self, index: int = 0) -> GeneratedProject:
        prompt = build_project_generation_prompt(self.prompt_template, self.config)
        payload = self._request_project(
            prompt=prompt,
            model=self.config.model.model_id,
            fallback_models=self.config.model.fallback_models,
            max_tokens=self.config.model.max_tokens,
            temperature=self.config.model.temperature,
            schema_version=f"generation-{index}",
        )
        project = canonicalize_project(GeneratedProject.model_validate(payload))
        validate_project(project, self.config.validation)
        return self.write_project(project)

    def repair_project(
        self, project: GeneratedProject, compile_manifest: CompileManifest, attempt: int = 1
    ) -> GeneratedProject:
        if not self.repair_prompt_template:
            raise ValueError("repair prompt template is not configured")
        prompt = build_project_repair_prompt(
            self.repair_prompt_template, project, compile_manifest
        )
        payload = self._request_project(
            prompt=prompt,
            model=self.config.model.repair_model_id or self.config.model.model_id,
            fallback_models=(
                self.config.model.repair_fallback_models or self.config.model.fallback_models
            ),
            max_tokens=self.config.model.repair_max_tokens or self.config.model.max_tokens,
            temperature=self.config.model.repair_temperature,
            schema_version=f"repair-{project.project_id}-{attempt}",
        )
        repaired = canonicalize_project(GeneratedProject.model_validate(payload))
        repaired = repaired.model_copy(update={"project_id": project.project_id})
        validate_project(repaired, self.config.validation)
        return self.rewrite_project(repaired)

    def _request_project(
        self,
        *,
        prompt: str,
        model: str,
        fallback_models: list[str],
        max_tokens: int,
        temperature: float,
        schema_version: str,
    ) -> dict[str, Any]:
        return self.client.generate_json(
            model=model,
            fallback_models=fallback_models,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=GeneratedProject.model_json_schema(),
            schema_version=schema_version,
        )

    def _claim_project_dir(self, base_id: str) -> tuple[str, Path]:
        """Atomically claim a unique project directory using mkdir as the claim step."""
        candidate_id = base_id
        suffix = 1
        while True:
            destination = self.project_root / candidate_id
            try:
                destination.mkdir(parents=True, exist_ok=False)
                return candidate_id, destination
            except FileExistsError:
                candidate_id = f"{base_id}_v{suffix}"
                suffix += 1

    def write_project(self, project: GeneratedProject) -> GeneratedProject:
        project_id, destination = self._claim_project_dir(project.project_id)
        if project_id != project.project_id:
            project = project.model_copy(update={"project_id": project_id})
        self._write_project_files(project, destination)
        self._write_manifest(project, destination)
        return project

    def rewrite_project(self, project: GeneratedProject) -> GeneratedProject:
        destination = self.project_root / project.project_id
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=False)
        self._write_project_files(project, destination)
        self._write_manifest(project, destination)
        return project

    def _write_project_files(self, project: GeneratedProject, destination: Path) -> None:
        for file in project.files:
            output = destination / file.path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(file.content, encoding="utf-8")

    def _write_manifest(self, project: GeneratedProject, destination: Path) -> None:
        manifest_path = destination / "project_manifest.json"
        manifest_json = json.dumps(project.model_dump(mode="python"), indent=2, sort_keys=True)
        manifest_path.write_text(manifest_json, encoding="utf-8")
        (self.manifest_root / f"{project.project_id}.json").write_text(
            manifest_json, encoding="utf-8"
        )

    @staticmethod
    def load_project(path: Path) -> GeneratedProject:
        return GeneratedProject.model_validate_json(path.read_text(encoding="utf-8"))
