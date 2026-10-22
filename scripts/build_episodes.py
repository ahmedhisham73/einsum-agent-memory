"""Build source-preserving episodes for the synthetic sample."""

import hashlib
import json
import runpy
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.contracts.json_output import parse_model_json

from packages.telemetry.token_ledger import TokenLedger


def load_module(relative_path):
    return runpy.run_path(str(ROOT / relative_path))


def validate_segments(segments, session):
    if not isinstance(segments, list) or not segments:
        raise ValueError("Expected a non-empty segments list")

    positions = {
        turn.turn_id: index
        for index, turn in enumerate(session.turns)
    }

    validated = []
    next_position = 0

    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Each segment must be an object")

        start_id = segment.get("start_turn_id")
        end_id = segment.get("end_turn_id")
        theme = segment.get("theme")

        if not isinstance(start_id, str) or not isinstance(end_id, str):
            raise ValueError("Boundary IDs must be strings")
        if start_id not in positions or end_id not in positions:
            raise ValueError("Unknown boundary turn ID")
        if not isinstance(theme, str) or not theme.strip():
            raise ValueError("Episode theme must not be empty")

        start = positions[start_id]
        end = positions[end_id]

        if start != next_position or end < start:
            raise ValueError("Episode gap, overlap or invalid ordering")

        validated.append((start, end, theme))
        next_position = end + 1

    if next_position != len(session.turns):
        raise ValueError("Episodes do not cover all session turns")

    return validated


def main():
    load_dotenv(ROOT / ".env", override=True)

    config = load_module(
        "services/llm-gateway/app/config.py"
    )["GatewayConfig"].from_env()

    complete_chat = load_module(
        "services/llm-gateway/app/providers/openai_compatible.py"
    )["complete_chat"]

    load_conversation = load_module(
        "services/dataset-service/app/ingestion/json_loader.py"
    )["load_conversation"]

    source_path = ROOT / "datasets/samples/architecture_history.json"
    prompt_path = ROOT / "prompts/ensimem/episode_partition.md"

    conversation = load_conversation(source_path)
    prompt = prompt_path.read_text(encoding="utf-8")
    experiment_id = "episode-build-" + uuid.uuid4().hex[:12]

    output_dir = ROOT / "outputs" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=False)

    ledger = TokenLedger()
    episodes = []
    traces = []
    status = "incomplete"

    try:
        for session in conversation.sessions:
            print("Partitioning:", session.session_id, flush=True)

            session_data = {
                "session_id": session.session_id,
                "observed_at": session.observed_at,
                "turns": [
                    {
                        "turn_id": turn.turn_id,
                        "speaker": turn.speaker,
                        "text": turn.text,
                    }
                    for turn in session.turns
                ],
            }

            result = complete_chat(
                config=config,
                ledger=ledger,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            session_data, ensure_ascii=False
                        ),
                    },
                ],
                experiment_id=experiment_id,
                approach="ensimem",
                phase="build",
                stage="episode_partitioning",
                provider="openai",
                max_completion_tokens=1200,
            )

            body = result["response"]
            traces.append({
                "session_id": session.session_id,
                "call_id": result["call_id"],
                "latency_seconds": result["request_latency_seconds"],
                "response": body,
            })

            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Model response did not finish normally")

            parsed = parse_model_json(choice["message"]["content"])
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object")

            segments = validate_segments(
                parsed.get("segments"), session
            )

            for number, (start, end, theme) in enumerate(segments, 1):
                episodes.append({
                    "episode_id": (
                        f"{conversation.conversation_id}::"
                        f"{session.session_id}::episode-{number:03d}"
                    ),
                    "conversation_id": conversation.conversation_id,
                    "session_id": session.session_id,
                    "observed_at": session.observed_at,
                    "theme": theme,
                    "turns": [
                        {
                            "turn_id": turn.turn_id,
                            "speaker": turn.speaker,
                            "text": turn.text,
                        }
                        for turn in session.turns[start:end + 1]
                    ],
                })

        status = "completed"

    finally:
        artifacts = {
            "manifest.json": {
                "experiment_id": experiment_id,
                "status": status,
                "source_kind": "synthetic",
                "requested_model": config.model,
                "dataset_sha256": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "prompt_sha256": hashlib.sha256(
                    prompt_path.read_bytes()
                ).hexdigest(),
                "episode_count": len(episodes),
            },
            "episodes.json": episodes,
            "traces.json": traces,
            "usage.json": ledger.export_records(),
            "token_summary.json": ledger.summary(experiment_id),
        }

        for filename, content in artifacts.items():
            (output_dir / filename).write_text(
                json.dumps(content, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        print("\nStatus:", status)
        print("Artifacts:", output_dir)
        print(json.dumps(ledger.summary(experiment_id), indent=2))

    print("\nValidated episodes:", len(episodes))


if __name__ == "__main__":
    main()
