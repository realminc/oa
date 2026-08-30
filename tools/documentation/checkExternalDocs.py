#!/usr/bin/env python3
"""Check OA's publishable documentation boundary and local links."""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.parse
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "external"
SDK_DOC_ASSETS = REPO / "sdk" / "asset" / "docs"
STATUS_EXEMPT = {DOCS / "pyPIReadme.md"}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
FORBIDDEN = {
    "docs" + "/internal": "private documentation path",
    "../internal": "relative private documentation path",
    ".cur" + "sor/": "private editor configuration",
    ".dev" + "in/": "private agent configuration",
    ".vs" + "code/": "private editor configuration",
    ".z" + "ed/": "private editor configuration",
    "tools" + "/release": "private release tooling",
}


def main() -> int:
    errors: list[str] = []
    markdown = sorted([
        *DOCS.rglob("*.md"),
        *SDK_DOC_ASSETS.rglob("*.md"),
    ])
    if not markdown:
        errors.append("docs/external: no public Markdown files found")

    for path in markdown:
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(REPO)

        if path not in STATUS_EXEMPT:
            header = "\n".join(text.splitlines()[:16]).lower()
            if "status:" not in header:
                errors.append(f"{relative}: missing Status in first 16 lines")

        for lineNumber, line in enumerate(text.splitlines(), 1):
            for rawTarget in MARKDOWN_LINK.findall(line):
                target = rawTarget.strip().split()[0].strip("<>")
                if not target or target.startswith(
                    ("#", "http://", "https://", "mailto:")
                ):
                    continue
                decoded = urllib.parse.unquote(target.split("#", 1)[0])
                if decoded and not (path.parent / decoded).resolve().exists():
                    errors.append(f"{relative}:{lineNumber}: broken link {decoded}")

        for fragment, reason in FORBIDDEN.items():
            if fragment in text:
                errors.append(f"{relative}: {reason}: {fragment}")
        if "/home/" in text or "/Users/" in text:
            errors.append(f"{relative}: contains a machine-local absolute path")

    generatedExamples = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/documentation/generateExamples.py"),
            "--check",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if generatedExamples.returncode != 0:
        detail = generatedExamples.stdout.strip() or generatedExamples.stderr.strip()
        errors.append(detail or "generated example inventory check failed")

    exampleTests = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/documentation/testGenerateExamples.py"),
            "-q",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if exampleTests.returncode != 0:
        detail = exampleTests.stdout.strip() or exampleTests.stderr.strip()
        errors.append(detail or "generated example inventory tests failed")

    if errors:
        print("\n".join(errors))
        print(f"[external-docs] FAIL: {len(errors)} problem(s)")
        return 1

    print(
        f"[external-docs] PASS: {len(markdown)} Markdown files, "
        "status/link/public-boundary/example checks clean"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
