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
    markdown = sorted(DOCS.rglob("*.md"))
    if not markdown:
        errors.append("docs/external: no public Markdown files found")

    for path in markdown:
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(REPO)

        if path not in STATUS_EXEMPT:
            header = "\n".join(text.splitlines()[:16]).lower()
            if "status:" not in header:
                errors.append(f"{relative}: missing Status in first 16 lines")

        for line_number, line in enumerate(text.splitlines(), 1):
            for raw_target in MARKDOWN_LINK.findall(line):
                target = raw_target.strip().split()[0].strip("<>")
                if not target or target.startswith(
                    ("#", "http://", "https://", "mailto:")
                ):
                    continue
                decoded = urllib.parse.unquote(target.split("#", 1)[0])
                if decoded and not (path.parent / decoded).resolve().exists():
                    errors.append(f"{relative}:{line_number}: broken link {decoded}")

        for fragment, reason in FORBIDDEN.items():
            if fragment in text:
                errors.append(f"{relative}: {reason}: {fragment}")
        if "/home/" in text or "/Users/" in text:
            errors.append(f"{relative}: contains a machine-local absolute path")

    generated_examples = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/documentation/generate_examples.py"),
            "--check",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if generated_examples.returncode != 0:
        detail = generated_examples.stdout.strip() or generated_examples.stderr.strip()
        errors.append(detail or "generated example inventory check failed")

    example_tests = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/documentation/test_generate_examples.py"),
            "-q",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if example_tests.returncode != 0:
        detail = example_tests.stdout.strip() or example_tests.stderr.strip()
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
