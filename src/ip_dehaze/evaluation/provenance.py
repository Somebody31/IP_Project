"""Run-record provenance helpers shared by every script."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path


def git_revision(cwd: Path | str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict[str, str]:
    import numpy
    import scipy

    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_cuda"] = str(torch.version.cuda)
    except ImportError:  # pragma: no cover - torch optional
        pass
    return versions
