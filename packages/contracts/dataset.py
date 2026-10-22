"""Conversation contracts preserving source text, time and media metadata."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Tuple


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class Turn:
    turn_id: str
    speaker: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        require_text(self.turn_id, "turn_id")
        require_text(self.speaker, "speaker")
        require_text(self.text, "text")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")


@dataclass(frozen=True)
class Session:
    session_id: str
    observed_at: str
    turns: Tuple[Turn, ...]
    source_timestamp: str = ""
    timezone_status: str = "specified"

    def __post_init__(self):
        require_text(self.session_id, "session_id")
        require_text(self.observed_at, "observed_at")

        parsed = datetime.fromisoformat(
            self.observed_at.replace("Z", "+00:00")
        )

        if self.timezone_status not in ("specified", "unspecified"):
            raise ValueError("Unsupported timezone_status")

        if self.timezone_status == "specified" and parsed.tzinfo is None:
            raise ValueError("Specified timezone requires an offset")

        if self.timezone_status == "unspecified":
            if parsed.tzinfo is not None:
                raise ValueError("Unspecified timezone must not invent an offset")
            require_text(self.source_timestamp, "source_timestamp")

        if not isinstance(self.turns, tuple) or not self.turns:
            raise ValueError("turns must be a non-empty tuple")

        if not all(isinstance(turn, Turn) for turn in self.turns):
            raise ValueError("Every session item must be a Turn")

        ids = [turn.turn_id for turn in self.turns]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate turn_id within session")


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    sessions: Tuple[Session, ...]

    def __post_init__(self):
        require_text(self.conversation_id, "conversation_id")

        if not isinstance(self.sessions, tuple) or not self.sessions:
            raise ValueError("sessions must be a non-empty tuple")

        if not all(isinstance(s, Session) for s in self.sessions):
            raise ValueError("Every conversation item must be a Session")

        ids = [session.session_id for session in self.sessions]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate session_id within conversation")
