"""Run offline correctness/UI tests and save a dated, source-linked result."""
import json
import platform
import sys
import unittest
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from time import perf_counter
from src.project_status import ROOT, source_fingerprint


def main():
    fingerprint = source_fingerprint()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    start = perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    packages = {}
    for name in ["streamlit", "langchain", "langgraph", "chromadb", "sentence-transformers"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": fingerprint,
        "source_unchanged_during_run": fingerprint == source_fingerprint(),
        "command": ".venv-agent/bin/python -m scripts.verify_project",
        "python": platform.python_version(), "packages": packages,
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
        "duration_seconds": round(perf_counter() - start, 3),
        "limitations": [
            "Offline tests use scripted models; no live Gemini quality score.",
            "Passing tests verifies checked behavior, not production readiness or root causes.",
        ],
    }
    path = ROOT / "reports/project_verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved:", path)
    return 0 if result.wasSuccessful() and report["source_unchanged_during_run"] else 1


if __name__ == "__main__":
    sys.exit(main())
