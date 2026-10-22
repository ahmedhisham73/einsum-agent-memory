"""Real API connectivity test; not an EnSIMem experiment."""

import json
import runpy
import sys
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.telemetry.token_ledger import TokenLedger


def main():
    load_dotenv(ROOT / ".env", override=True)

    config_module = runpy.run_path(
        str(ROOT / "services/llm-gateway/app/config.py")
    )
    provider_module = runpy.run_path(
        str(ROOT / "services/llm-gateway/app/providers/openai_compatible.py")
    )

    config = config_module["GatewayConfig"].from_env()
    ledger = TokenLedger()

    print("Sending a real API request...")
    print("Model:", config.model)

    result = provider_module["complete_chat"](
        config=config,
        ledger=ledger,
        messages=[{
            "role": "user",
            "content": "Reply with exactly: Connected",
        }],
        experiment_id="connectivity-test-only",
        approach="ensimem",
        phase="query",
        stage="connectivity_check",
        provider=urlsplit(config.base_url).hostname,
        query_id="connectivity-001",
        max_completion_tokens=128,
    )

    choices = result["response"].get("choices") or []
    message = choices[0].get("message", {}) if choices else {}

    print("\nAPI request completed.")
    print("Reply:", message.get("content"))
    print(
        "Request latency:",
        round(result["request_latency_seconds"], 3),
        "seconds",
    )
    print("\nToken usage:")
    print(json.dumps(
        ledger.summary("connectivity-test-only"),
        indent=2,
    ))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        sys.exit(1)
