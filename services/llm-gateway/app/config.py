"""Load LLM gateway configuration without exposing credentials."""

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class GatewayConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls):
        base_url = os.environ.get("LLM_BASE_URL", "").strip()
        model = os.environ.get("LLM_MODEL", "").strip()
        api_key = os.environ.get("LLM_API_KEY", "").strip()

        required = {
            "LLM_BASE_URL": base_url,
            "LLM_MODEL": model,
            "LLM_API_KEY": api_key,
        }

        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "Missing environment variables: " + ", ".join(missing)
            )

        parsed = urlsplit(base_url)

        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("LLM_BASE_URL must be an HTTP(S) URL")

        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "LLM_BASE_URL must not contain credentials, query or fragment"
            )

        try:
            timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
        except ValueError:
            raise ValueError("LLM_TIMEOUT_SECONDS must be a number") from None

        if not 0 < timeout <= 600:
            raise ValueError("LLM_TIMEOUT_SECONDS must be between 0 and 600")

        return cls(
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key,
            timeout_seconds=timeout,
        )
