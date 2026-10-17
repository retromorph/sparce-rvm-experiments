"""Legacy-compatible synthetic-only smoke. No external LLM or API verifier."""
from sparce.smoke import run_smoke

if __name__ == "__main__":
    import json

    print(json.dumps(run_smoke(), indent=2))
