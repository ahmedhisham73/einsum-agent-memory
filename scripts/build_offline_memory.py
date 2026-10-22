"""Build source-preserving offline memory from a LoCoMo conversation."""

import argparse
import re
import hashlib
import json
import runpy
import sys
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.contracts.json_output import parse_model_json

from packages.contracts.dataset import Conversation, Session, Turn
from packages.telemetry.token_ledger import TokenLedger


def module(path):
    return runpy.run_path(str(ROOT / path))


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default="conv-26")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=True)

    config = module(
        "services/llm-gateway/app/config.py"
    )["GatewayConfig"].from_env()

    complete_chat = module(
        "services/llm-gateway/app/providers/openai_compatible.py"
    )["complete_chat"]

    validate_segments = module(
        "scripts/build_episodes.py"
    )["validate_segments"]

    validate_records = module(
        "scripts/build_entity_index.py"
    )["validate_records"]

    partition_prompt = (
        ROOT / "prompts/ensimem/episode_partition.md"
    ).read_text(encoding="utf-8")

    extraction_prompt = (
        ROOT / "prompts/ensimem/index_extraction.md"
    ).read_text(encoding="utf-8")

    media_rules = """
Additional source-handling rules:
- Preserve the identity of each source turn.
- Captions and image search queries are auxiliary textual observations,
  not verified user statements. Preserve their uncertainty and origin.
- Image URLs are opaque provenance. Do not fetch them or infer their content.
- Ignore operational metadata such as re-download when extracting facts.
- Do not invent a timezone or a precise event time.
"""
    partition_prompt += media_rules
    extraction_prompt += media_rules

    source_path = ROOT / "datasets/raw/locomo/locomo10.json"
    source_bytes = source_path.read_bytes()
    dataset = json.loads(source_bytes)

    matches = [
        item for item in dataset
        if item["sample_id"] == args.sample_id
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching conversation")

    # Read only conversation data. QA and generated summaries are excluded.
    source = matches[0]["conversation"]

    session_numbers = sorted(
        int(key.split("_")[1])
        for key in source
        if re.fullmatch(r"session_\d+", key)
    )

    sessions = []

    for number in session_numbers:
        key = f"session_{number}"
        timestamp = source[f"{key}_date_time"]

        sessions.append(Session(
            session_id=key,
            observed_at=datetime.strptime(
                timestamp.strip(), "%I:%M %p on %d %B, %Y"
            ).isoformat(),
            source_timestamp=timestamp,
            timezone_status="unspecified",
            turns=tuple(
                Turn(
                    turn_id=item["dia_id"],
                    speaker=item["speaker"],
                    text=item["text"],
                    metadata={
                        name: value
                        for name, value in item.items()
                        if name not in ("dia_id", "speaker", "text")
                    },
                )
                for item in source[key]
            ),
        ))

    conversation = Conversation(
        conversation_id=args.sample_id,
        sessions=tuple(sessions),
    )

    run_id = "offline-memory-" + uuid.uuid4().hex[:12]
    output_dir = ROOT / "outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    ledger = TokenLedger()
    episodes = []
    records = []
    traces = []

    manifest = {
        "experiment_id": run_id,
        "status": "incomplete",
        "stage": "partitioning",
        "source_kind": "locomo_benchmark",
        "conversation_id": conversation.conversation_id,
        "session_count": len(sessions),
        "raw_dataset_sha256": sha256(source_bytes),
        "partition_prompt_sha256": sha256(partition_prompt.encode()),
        "extraction_prompt_sha256": sha256(extraction_prompt.encode()),
        "requested_model": config.model,
        "timezone_policy": "Source timezone unspecified",
        "media_policy": "Textual metadata retained; no image URLs fetched",
    }

    def checkpoint():
        manifest["episode_count"] = len(episodes)
        manifest["record_count"] = len(records)

        for filename, value in {
            "manifest.json": manifest,
            "episodes.json": episodes,
            "records.json": records,
            "traces.json": traces,
            "usage.json": ledger.export_records(),
            "token_summary.json": ledger.summary(run_id),
        }.items():
            save_json(output_dir / filename, value)

    def ask(prompt, payload, stage, source_id):
        result = complete_chat(
            config=config,
            ledger=ledger,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            experiment_id=run_id,
            approach="ensimem",
            phase="build",
            stage=stage,
            provider="openai",
            max_completion_tokens=8000,
        )

        body = result["response"]
        traces.append({
            "source_id": source_id,
            "stage": stage,
            "call_id": result["call_id"],
            "latency_seconds": result["request_latency_seconds"],
            "response": body,
        })

        choice = body["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Incomplete model response; inspect traces.json")

        parsed = parse_model_json(choice["message"]["content"])
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")
        return parsed

    save_json(
        output_dir / "source_conversation.json",
        asdict(conversation),
    )

    print("Run:", run_id)
    print("Conversation:", conversation.conversation_id)
    print("Sessions:", len(sessions))
    print("Output:", output_dir, flush=True)

    try:
        for position, session in enumerate(sessions, 1):
            print(
                f"Partitioning session {position}/{len(sessions)}",
                flush=True,
            )

            parsed = ask(
                partition_prompt,
                asdict(session),
                "episode_partitioning",
                session.session_id,
            )

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
                    "source_timestamp": session.source_timestamp,
                    "timezone_status": session.timezone_status,
                    "theme": theme,
                    "turns": [
                        asdict(turn)
                        for turn in session.turns[start:end + 1]
                    ],
                })

            checkpoint()

        manifest["stage"] = "extraction"

        for position, episode in enumerate(episodes, 1):
            print(
                f"Extracting episode {position}/{len(episodes)}",
                flush=True,
            )

            # Exclude generated theme from answer-bearing evidence.
            payload = {
                key: value
                for key, value in episode.items()
                if key != "theme"
            }

            parsed = ask(
                extraction_prompt,
                payload,
                "entity_property_extraction",
                episode["episode_id"],
            )

            extracted = validate_records(
                parsed.get("records"), episode
            )

            for item in extracted:
                records.append({
                    **item,
                    "record_id": f"{run_id}::r{len(records) + 1:06d}",
                    "episode_id": episode["episode_id"],
                    "conversation_id": episode["conversation_id"],
                    "session_id": episode["session_id"],
                    "observed_at": episode["observed_at"],
                    "source_timestamp": episode["source_timestamp"],
                    "timezone_status": episode["timezone_status"],
                })

            checkpoint()

        manifest["status"] = "completed"
        manifest["stage"] = "finished"

    finally:
        checkpoint()
        print("\nStatus:", manifest["status"])
        print("Episodes:", len(episodes))
        print("Records:", len(records))
        print("Artifacts:", output_dir)
        print(json.dumps(ledger.summary(run_id), indent=2))


if __name__ == "__main__":
    main()
