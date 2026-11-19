from pathlib import Path

from research_platform.backend.app.database_backup import (
    build_mysqldump_command,
    build_restore_command,
    create_backup_manifest,
    validate_backup_manifest,
)


def test_backup_commands_never_include_password() -> None:
    url = "mysql+pymysql://user:secret@example.test:3307/platform"
    dump = build_mysqldump_command(url, "backup.sql")
    restore = build_restore_command(url, "backup.sql")
    assert "secret" not in " ".join(dump + restore)
    assert "platform" in dump
    assert ">" in dump and "<" in restore


def test_backup_manifest_round_trip(tmp_path: Path) -> None:
    backup = tmp_path / "backup.sql"
    backup.write_bytes(b"CREATE TABLE audit_events(id varchar(36));\n")
    manifest = create_backup_manifest("mysql://user:secret@localhost/platform", backup)
    assert validate_backup_manifest(manifest, backup)
    backup.write_bytes(backup.read_bytes() + b"changed")
    assert not validate_backup_manifest(manifest, backup)

