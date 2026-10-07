#!/usr/bin/env python3
"""
Apply the RAGAS 0.4.3 VertexAI import fix described in the project README.

RAGAS 0.4.3 imports ChatVertexAI and VertexAI from old langchain-community
paths that no longer exist, so `import ragas` fails. This script rewrites
those two imports in the installed ragas/llms/base.py to use
langchain-google-vertexai instead. It is safe to run more than once.

Usage:
    python fix_ragas_imports.py
"""

import importlib.util
import sys
from pathlib import Path

REPLACEMENTS = {
    "from langchain_community.chat_models.vertexai import ChatVertexAI":
        "from langchain_google_vertexai import ChatVertexAI",
    "from langchain_community.llms import VertexAI":
        "from langchain_google_vertexai import VertexAI",
}


def main() -> int:
    spec = importlib.util.find_spec("ragas")
    if spec is None or not spec.submodule_search_locations:
        print("RAGAS is not installed. Run: pip install -r requirements.txt")
        return 1

    base_file = Path(list(spec.submodule_search_locations)[0]) / "llms" / "base.py"
    if not base_file.exists():
        print(f"Could not find {base_file}")
        return 1

    source = base_file.read_text(encoding="utf-8")
    patched = source
    for old, new in REPLACEMENTS.items():
        patched = patched.replace(old, new)

    if patched == source:
        print(f"No change needed: {base_file} already uses langchain_google_vertexai imports.")
    else:
        base_file.write_text(patched, encoding="utf-8")
        print(f"Patched VertexAI imports in {base_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
