"""Read reproducible verification artifacts; never equate tests with AI accuracy."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_fingerprint(root=ROOT):
    patterns = ["app.py", "src/*.py", "tests/*.py", "tests/*.json",
                "scripts/verify_project.py", "requirements*.txt", ".streamlit/config.toml"]
    paths = sorted({p for pattern in patterns for p in root.glob(pattern)})
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_report(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def verification_status(root=ROOT):
    report = read_report(root / "reports/project_verification.json")
    if not report:
        return None, False
    return report, (report.get("source_sha256") == source_fingerprint(root)
                    and report.get("source_unchanged_during_run") is True)
