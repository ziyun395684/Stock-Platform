from pathlib import Path

from research_platform.backend.app.secret_hygiene import scan_secret_hygiene


def test_secret_hygiene_ignores_example_and_detects_private_material(tmp_path: Path) -> None:
    (tmp_path / "safe.txt").write_text("DATABASE_URL=mysql://user:placeholder@host/db", encoding="utf-8")
    assert scan_secret_hygiene(tmp_path) == []
    (tmp_path / ".env").write_text("DATABASE_PASSWORD=real", encoding="utf-8")
    findings = scan_secret_hygiene(tmp_path)
    assert any("secret-like file" in item for item in findings)
