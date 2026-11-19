from __future__ import annotations

import sys
from pathlib import Path

from research_platform.scripts import database_backup


def test_backup_cli_defaults_to_dry_run(monkeypatch, tmp_path: Path, capsys) -> None:
    target = tmp_path / "platform.sql"
    monkeypatch.setattr(sys, "argv", ["database_backup", "backup", str(target), "--database-url", "mysql://user:secret@db/platform"])
    assert database_backup.main() == 0
    assert not target.exists()
    assert "secret" not in capsys.readouterr().out

