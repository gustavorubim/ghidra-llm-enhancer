from __future__ import annotations

import json
from pathlib import Path

from decomp_clarifier.schemas.compiler import CompileManifest
from decomp_clarifier.schemas.generation import GeneratedProject
from decomp_clarifier.schemas.model_io import PromptInput
from decomp_clarifier.settings import GenerationConfig


def load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_project_generation_prompt(template: str, config: GenerationConfig) -> str:
    return template.format(
        topics=json.dumps(config.generation.topic_weights, indent=2, sort_keys=True),
        difficulty_weights=json.dumps(
            config.generation.difficulty_weights, indent=2, sort_keys=True
        ),
        validation_rules=json.dumps(
            config.validation.model_dump(mode="python"), indent=2, sort_keys=True
        ),
    )


def build_cleanup_prompt(template: str, prompt_input: PromptInput) -> str:
    sections = [
        template.strip(),
        "",
        f"Task type: {prompt_input.task_type}",
        "",
        "Decompiler:",
        f"<code>\n{prompt_input.decompiled_code}\n</code>",
        "",
        "Assembly:",
        f"<asm>\n{prompt_input.assembly}\n</asm>",
        "",
        f"Strings: {json.dumps(prompt_input.strings)}",
        f"Imports: {json.dumps(prompt_input.imports)}",
        f"Callers: {json.dumps(prompt_input.callers)}",
        f"Callees: {json.dumps(prompt_input.callees)}",
    ]
    if prompt_input.semantic_summary:
        sections.extend(["", f"Semantic summary hint: {prompt_input.semantic_summary}"])
    return "\n".join(sections).strip()


def build_project_repair_prompt(
    template: str, project: GeneratedProject, compile_manifest: CompileManifest
) -> str:
    failed_tests = [
        {
            "name": result.name,
            "returncode": result.returncode,
            "expected": next(
                (test.expected for test in project.tests if test.name == result.name),
                "",
            ),
            "actual_stdout": result.stdout,
            "actual_stderr": result.stderr,
        }
        for result in compile_manifest.test_results
        if not result.passed
    ]
    payload = {
        "project": project.model_dump(mode="python"),
        "compile": {
            "compiler_family": compile_manifest.compiler_family,
            "compiler_version": compile_manifest.compiler_version,
            "host_os": compile_manifest.host_os,
            "opt_level": compile_manifest.opt_level,
            "build_log": compile_manifest.build_log,
            "binary_count": len(compile_manifest.binaries),
            "failed_tests": failed_tests,
        },
    }
    return "\n".join(
        [
            template.strip(),
            "",
            "Current project and failure context:",
            json.dumps(payload, indent=2, sort_keys=True),
        ]
    ).strip()
