"""Load normalized conversations with source time and media metadata."""

import json
from pathlib import Path

from packages.contracts.dataset import Conversation, Session, Turn


def load_conversation(path) -> Conversation:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Dataset must be a JSON object")

    if data.get("schema_version") != "1.0":
        raise ValueError("Unsupported dataset schema_version")

    try:
        sessions = []

        for session_data in data["sessions"]:
            turns = tuple(
                Turn(
                    turn_id=turn["turn_id"],
                    speaker=turn["speaker"],
                    text=turn["text"],
                    metadata=turn.get("metadata", {}),
                )
                for turn in session_data["turns"]
            )

            sessions.append(Session(
                session_id=session_data["session_id"],
                observed_at=session_data["observed_at"],
                turns=turns,
                source_timestamp=session_data.get("source_timestamp", ""),
                timezone_status=session_data.get(
                    "timezone_status", "specified"
                ),
            ))

        return Conversation(
            conversation_id=data["conversation_id"],
            sessions=tuple(sessions),
        )

    except (KeyError, TypeError) as error:
        raise ValueError(
            "Dataset structure does not match the conversation schema"
        ) from error
