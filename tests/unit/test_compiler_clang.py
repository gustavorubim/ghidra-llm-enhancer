from __future__ import annotations

from pathlib import Path

import pytest

from decomp_clarifier.adapters import compiler_clang
from decomp_clarifier.adapters.compiler_clang import (
    ClangCompiler,
    compiler_not_found_message,
    resolve_clang_executable,
)
from decomp_clarifier.settings import CompilerProfile


def test_resolve_clang_uses_explicit_path(tmp_path: Path) -> None:
    fake_clang = tmp_path / "clang.exe"
    fake_clang.write_text("", encoding="utf-8")

    assert resolve_clang_executable(str(fake_clang)) == str(fake_clang.resolve())
    assert ClangCompiler(CompilerProfile(executable=str(fake_clang))).executable == str(
        fake_clang.resolve()
    )


def test_resolve_clang_probes_visual_studio_path_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clang_dir = (
        tmp_path
        / "Microsoft Visual Studio"
        / "2022"
        / "Community"
        / "VC"
        / "Tools"
        / "Llvm"
        / "x64"
        / "bin"
    )
    clang_dir.mkdir(parents=True)
    fake_clang = clang_dir / "clang.exe"
    fake_clang.write_text("", encoding="utf-8")

    monkeypatch.setattr(compiler_clang.os, "name", "nt", raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.delenv("LocalAppData", raising=False)
    monkeypatch.setattr(compiler_clang, "which", lambda command: None)

    assert resolve_clang_executable("clang") == str(fake_clang.resolve())


def test_clang_compiler_error_mentions_override() -> None:
    compiler = ClangCompiler(CompilerProfile(executable="missing-clang"))

    with pytest.raises(FileNotFoundError, match="DECOMP_CLARIFIER_COMPILER_EXECUTABLE"):
        _ = compiler.executable

    assert "configs/compile/*.yaml" in compiler_not_found_message("missing-clang")


def test_build_command_adds_crt_define_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    compiler = ClangCompiler(CompilerProfile(executable="clang", extra_flags=["-Wall"]))
    monkeypatch.setattr(compiler_clang.os, "name", "nt", raising=False)
    compiler.__dict__["executable"] = "clang.exe"

    command = compiler.build_command([Path("main.c")], Path("out.exe"))

    assert "-D_CRT_SECURE_NO_WARNINGS" in command
    assert "-Wall" in command
