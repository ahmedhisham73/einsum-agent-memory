"""Recover all existing offline builds, reusing saved model responses."""

import hashlib
import json
import runpy
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.contracts.dataset import Session, Turn
from packages.contracts.json_output import parse_model_json
from packages.contracts.usage import TokenUsage
from packages.telemetry.token_ledger import TokenLedger

MEDIA_RULES = """
Additional source-handling rules:
- Preserve the identity of each source turn.
- Captions and image search queries are auxiliary textual observations,
  not verified user statements. Preserve their uncertainty and origin.
- Image URLs are opaque provenance. Do not fetch them or infer their content.
- Ignore operational metadata such as re-download when extracting facts.
- Do not invent a timezone or a precise event time.
"""


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    load_dotenv(ROOT / ".env", override=True)

    config = runpy.run_path(
        str(ROOT / "services/llm-gateway/app/config.py")
    )["GatewayConfig"].from_env()
    config = replace(config, timeout_seconds=180)

    complete_chat = runpy.run_path(
        str(ROOT / "services/llm-gateway/app/providers/openai_compatible.py")
    )["complete_chat"]

    validate_segments = runpy.run_path(
        str(ROOT / "scripts/build_episodes.py")
    )["validate_segments"]

    validate_records = runpy.run_path(
        str(ROOT / "scripts/build_entity_index.py")
    )["validate_records"]

    partition_prompt = (
        ROOT / "prompts/ensimem/episode_partition.md"
    ).read_text() + MEDIA_RULES
    extraction_prompt = (
        ROOT / "prompts/ensimem/index_extraction.md"
    ).read_text() + MEDIA_RULES

    raw_path = ROOT / "datasets/raw/locomo/locomo10.json"
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    sample_ids = [sample["sample_id"] for sample in read(raw_path)]
    state_path = ROOT / "outputs/offline-recovery-v11.json"
    results = read(state_path) if state_path.exists() else {}

    def recover(folder):
        manifest = read(folder / "manifest.json")

        if manifest["raw_dataset_sha256"] != raw_hash:
            raise ValueError("Dataset hash changed")
        if manifest["requested_model"] != config.model:
            raise ValueError("Model differs from original build")
        if digest(partition_prompt) != manifest["partition_prompt_sha256"]:
            raise ValueError("Partition prompt differs from original build")
        if digest(extraction_prompt) != manifest["extraction_prompt_sha256"]:
            raise ValueError("Extraction prompt differs from original build")

        source = read(folder / "source_conversation.json")
        traces = read(folder / "traces.json")
        old_usage = read(folder / "usage.json")

        backup = folder / (
            "recovery-backup-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        )
        backup.mkdir()
        for filename in (
            "manifest.json", "episodes.json", "records.json",
            "traces.json", "usage.json", "token_summary.json",
            "normalizations.json",
        ):
            path = folder / filename
            if path.exists():
                shutil.copy2(path, backup / filename)

        ledger = TokenLedger()
        for item in old_usage:
            ledger.record(TokenUsage(**item))

        cached = {}
        for trace in traces:
            source_id = trace.get("source_id", trace.get("episode_id"))
            if source_id:
                cached[(trace["stage"], source_id)] = trace

        episodes = []
        records = []
        normalizations = []
        reused = 0
        new_calls = 0
        run_id = manifest["experiment_id"]

        manifest["status"] = "incomplete"
        manifest["recovery_version"] = "v1.1"
        manifest.pop("recovery_error", None)

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
                "normalizations.json": normalizations,
            }.items():
                save(folder / filename, value)

        def obtain(prompt, payload, stage, source_id):
            nonlocal reused, new_calls

            trace = cached.get((stage, source_id))
            if trace is not None:
                choice = trace["response"]["choices"][0]
                if choice.get("finish_reason") == "stop":
                    parsed = parse_model_json(choice["message"]["content"])
                    reused += 1
                    return parsed

            # At most two attempts, only retrying transient request failures.
            for attempt in (1, 2, 3, 4):
                try:
                    new_calls += 1
                    result = complete_chat(
                        config=config,
                        ledger=ledger,
                        messages=[
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": json.dumps(
                                    payload, ensure_ascii=False
                                ),
                            },
                        ],
                        experiment_id=run_id,
                        approach="ensimem",
                        phase="build",
                        stage=stage,
                        provider="openai",
                        max_completion_tokens=8000,
                        attempt=attempt,
                    )
                    break
                except RuntimeError as error:
                    message = str(error)
                    transient = (
                        any(
                            marker in message
                            for marker in (
                                "HTTP=429", "HTTP=500", "HTTP=502",
                                "HTTP=503", "HTTP=504",
                                "TimeoutError", "timed out",
                                "ConnectionResetError",
                                "nodename nor servname",
                                "Name or service not known",
                                "Temporary failure in name resolution",
                            )
                        )
                        and "insufficient_quota" not in message
                    )
                    checkpoint()
                    if attempt == 4 or not transient:
                        raise
                    print("  Transient request failure; one retry.", flush=True)
                    time.sleep(min(3 * 2 ** (attempt - 1), 24))

            body = result["response"]
            trace = {
                "source_id": source_id,
                "stage": stage,
                "call_id": result["call_id"],
                "latency_seconds": result["request_latency_seconds"],
                "response": body,
            }
            traces.append(trace)
            cached[(stage, source_id)] = trace

            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError(
                    f"Incomplete model response: {source_id}; "
                    "raw response preserved"
                )
            return parse_model_json(choice["message"]["content"])

        try:
            manifest["stage"] = "partitioning"

            for number, item in enumerate(source["sessions"], 1):
                session = Session(
                    session_id=item["session_id"],
                    observed_at=item["observed_at"],
                    source_timestamp=item.get("source_timestamp", ""),
                    timezone_status=item.get("timezone_status", "specified"),
                    turns=tuple(Turn(**turn) for turn in item["turns"]),
                )

                print(
                    f'  Session {number}/{len(source["sessions"])}',
                    flush=True,
                )
                parsed = obtain(
                    partition_prompt, asdict(session),
                    "episode_partitioning", session.session_id,
                )
                segments = validate_segments(
                    parsed.get("segments"), session
                )

                for index, (start, end, theme) in enumerate(segments, 1):
                    episodes.append({
                        "episode_id": (
                            f'{source["conversation_id"]}::'
                            f"{session.session_id}::episode-{index:03d}"
                        ),
                        "conversation_id": source["conversation_id"],
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

            for number, episode in enumerate(episodes, 1):
                print(
                    f"  Episode {number}/{len(episodes)}",
                    flush=True,
                )
                parsed = obtain(
                    extraction_prompt,
                    {
                        key: value for key, value in episode.items()
                        if key != "theme"
                    },
                    "entity_property_extraction",
                    episode["episode_id"],
                )
                extracted = validate_records(
                    parsed.get("records"), episode
                )

                for item in extracted:
                    record = {
                        **item,
                        "record_id": f"{run_id}::r{len(records) + 1:06d}",
                        "episode_id": episode["episode_id"],
                        "conversation_id": episode["conversation_id"],
                        "session_id": episode["session_id"],
                        "observed_at": episode["observed_at"],
                        "source_timestamp": episode["source_timestamp"],
                        "timezone_status": episode["timezone_status"],
                    }
                    records.append(record)

                    if "entity_type_original" in item:
                        normalizations.append({
                            "record_id": record["record_id"],
                            "field": "entity_type",
                            "from": item["entity_type_original"],
                            "to": item["entity_type"],
                        })
                checkpoint()

            # Check that source turns survived partitioning exactly.
            original = [
                (session["session_id"], turn)
                for session in source["sessions"]
                for turn in session["turns"]
            ]
            restored = [
                (episode["session_id"], turn)
                for episode in episodes
                for turn in episode["turns"]
            ]
            if original != restored:
                raise ValueError("Source coverage or content mismatch")

            manifest["status"] = "completed"
            manifest["stage"] = "finished"

        except Exception as error:
            manifest["recovery_error"] = str(error)
            raise
        finally:
            checkpoint()
            print(
                f'  Status: {manifest["status"]} | '
                f"Cached responses reused: {reused} | "
                f"New API attempts: {new_calls}",
                flush=True,
            )

    for sample_id in sample_ids:
        candidates = []

        for folder in (ROOT / "outputs").glob("offline-memory-*"):
            if "-backup-" in folder.name:
                continue
            path = folder / "manifest.json"
            if not path.exists():
                continue
            manifest = read(path)
            if (
                manifest.get("conversation_id") == sample_id
                and manifest.get("raw_dataset_sha256") == raw_hash
            ):
                candidates.append((folder, manifest))

        completed = [
            item for item in candidates
            if item[1]["status"] == "completed"
        ]

        if completed:
            folder, manifest = completed[0]
            results[sample_id] = {
                "status": "completed",
                "memory": str(folder),
                "reused_completed_build": True,
            }
            print(f"\n{sample_id}: already completed; skipped.", flush=True)
            continue

        if not candidates:
            results[sample_id] = {
                "status": "missing",
                "error": "No existing build found",
            }
            print(f"\n{sample_id}: no saved build found.", flush=True)
            continue

        folder, _ = max(
            candidates,
            key=lambda item: (item[0] / "manifest.json").stat().st_mtime,
        )
        print(f"\nRECOVERING {sample_id}: {folder.name}", flush=True)

        try:
            recover(folder)
            results[sample_id] = {
                "status": "completed", "memory": str(folder)
            }
        except Exception as error:
            results[sample_id] = {
                "status": "incomplete",
                "memory": str(folder),
                "error": str(error),
            }
            print("  ERROR:", error, flush=True)

        save(ROOT / "outputs/offline-recovery-v11.json", results)

    save(ROOT / "outputs/offline-recovery-v11.json", results)
    print("\n=== OFFLINE RECOVERY SUMMARY ===")
    for sample_id, item in results.items():
        print(sample_id, ":", item["status"])
        if item.get("error"):
            print(" ", item["error"])


if __name__ == "__main__":
    main()
