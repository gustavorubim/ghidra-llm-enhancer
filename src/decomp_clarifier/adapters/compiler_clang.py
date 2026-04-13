from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

from decomp_clarifier.adapters.subprocess_utils import run_subprocess, which
from decomp_clarifier.settings import CompilerProfile


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        normalized = path.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _windows_candidate_names(command: str) -> list[str]:
    path = Path(command)
    names = [path.name]
    if path.suffix.lower() != ".exe":
        names.append(f"{path.name}.exe")
    return list(dict.fromkeys(names))


def _explicit_path_candidates(command: str) -> list[Path]:
    path = Path(command)
    if path.is_absolute() or path.parent != Path("."):
        candidates = [path]
        if os.name == "nt" and path.suffix.lower() != ".exe":
            candidates.append(path.with_suffix(".exe"))
        return candidates
    return []


def _windows_probe_directories() -> list[Path]:
    directories: list[Path] = []
    fixed_roots = (
        ("ProgramFiles", Path("LLVM") / "bin"),
        ("ProgramFiles(x86)", Path("LLVM") / "bin"),
        ("LocalAppData", Path("Programs") / "LLVM" / "bin"),
    )
    for env_var, suffix in fixed_roots:
        root = os.getenv(env_var)
        if root:
            candidate = Path(root) / suffix
            if candidate.is_dir():
                directories.append(candidate)

    visual_studio_patterns = (
        "*/*/VC/Tools/Llvm/x64/bin",
        "*/*/VC/Tools/Llvm/bin",
        "*/*/VC/Tools/Llvm/ARM64/bin",
    )
    for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.getenv(env_var)
        if not root:
            continue
        vs_root = Path(root) / "Microsoft Visual Studio"
        if not vs_root.is_dir():
            continue
        for pattern in visual_studio_patterns:
            directories.extend(path for path in vs_root.glob(pattern) if path.is_dir())
    return _dedupe_paths(directories)


def resolve_clang_executable(command: str) -> str | None:
    for candidate in _explicit_path_candidates(command):
        if candidate.is_file():
            return str(candidate.resolve(strict=False))

    resolved = which(command)
    if resolved is not None:
        return resolved

    if os.name != "nt":
        return None

    candidate_names = _windows_candidate_names(command)
    for directory in _windows_probe_directories():
        for name in candidate_names:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate.resolve(strict=False))
    return None


def compiler_not_found_message(command: str) -> str:
    return (
        f"compiler not found: {command}. Install LLVM/Clang and add it to PATH, "
        "set DECOMP_CLARIFIER_COMPILER_EXECUTABLE in your environment or .env, "
        "or set compiler.executable in configs/compile/*.yaml."
    )


class ClangCompiler:
    def __init__(self, profile: CompilerProfile) -> None:
        self.profile = profile

    @cached_property
    def executable(self) -> str:
        resolved = resolve_clang_executable(self.profile.executable)
        if resolved is None:
            raise FileNotFoundError(compiler_not_found_message(self.profile.executable))
        return resolved

    def version(self) -> str:
        result = run_subprocess([self.executable, "--version"])
        result.raise_for_error()
        return result.stdout.splitlines()[0] if result.stdout else "unknown"

    def build_command(self, sources: list[Path], output_path: Path) -> list[str]:
        args = [
            self.executable,
            f"-std={self.profile.c_standard}",
            f"-{self.profile.opt_level}",
        ]
        if os.name == "nt":
            args.append("-D_CRT_SECURE_NO_WARNINGS")
        args.extend(self.profile.extra_flags)
        if self.profile.warnings_as_errors:
            args.append("-Werror")
        args.extend(str(path) for path in sources)
        args.extend(["-o", str(output_path)])
        return args

    def compile(
        self, sources: list[Path], output_path: Path, cwd: Path
    ) -> tuple[list[str], str, str, int]:
        command = self.build_command(sources, output_path)
        result = run_subprocess(command, cwd=cwd)
        return command, result.stdout, result.stderr, result.returncode
