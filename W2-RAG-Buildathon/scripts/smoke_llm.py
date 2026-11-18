"""Live smoke test for the watsonx.ai GPT-OSS-120B connection (uses .env; never prints the token).

Run from the project root:  python scripts/smoke_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.rag_service import build_llm  # noqa: E402
from config.settings import get_settings  # noqa: E402


def main() -> int:
    s = get_settings()
    problems = s.validate()
    if problems:
        print("Configuration problems:\n - " + "\n - ".join(problems))
        return 1
    print(f"Model:    {s.llm_model}\nBase URL: {s.llm_base_url}\nToken:    set ({len(s.watsonx_token)} chars)")
    try:
        reply = build_llm(s).invoke("Reply with exactly: watsonx OK")
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 2
    print(f"Reply:    {reply.content!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
