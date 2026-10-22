"""Shared token-usage contract for every model API call."""

from dataclasses import dataclass
from typing import Literal, Optional

Approach = Literal["agentic_rag", "ensimem"]
Phase = Literal["build", "query", "evaluation"]
UsageKind = Literal["generation", "embedding"]


@dataclass(frozen=True)
class TokenUsage:
    # Identity and attribution
    call_id: str
    experiment_id: str
    approach: Approach
    phase: Phase
    stage: str
    provider: str
    model: str

    # None means the provider did not report this value.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None

    query_id: Optional[str] = None
    usage_kind: UsageKind = "generation"
    attempt: int = 1

    def __post_init__(self):
        if self.approach not in ("agentic_rag", "ensimem"):
            raise ValueError("Unsupported approach")

        if self.phase not in ("build", "query", "evaluation"):
            raise ValueError("Unsupported phase")

        if self.usage_kind not in ("generation", "embedding"):
            raise ValueError("Unsupported usage kind")

        for name in (
            "call_id", "experiment_id", "stage", "provider", "model"
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")

        for name in (
            "input_tokens", "output_tokens", "cached_input_tokens"
        ):
            value = getattr(self, name)
            if value is not None:
                if type(value) is not int or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer")

        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")

        if self.phase == "query" and not self.query_id:
            raise ValueError("Query usage requires query_id")

        if (
            self.input_tokens is not None
            and self.cached_input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ValueError("Cached tokens cannot exceed input tokens")

    @property
    def total_tokens(self) -> Optional[int]:
        """Cached input is already included in input_tokens."""
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens
