"""Resume extraction from saved episodes, traces and usage."""

import hashlib
import json
import runpy
import shutil
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.contracts.json_output import parse_model_json

from packages.contracts.usage import TokenUsage
from packages.telemetry.token_ledger import TokenLedger


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(content, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/resume_offline_memory.py RUN_FOLDER"
        )

    folder = Path(sys.argv[1]).resolve()
    manifest = read_json(folder / "manifest.json")
    episodes = read_json(folder / "episodes.json")
    traces = read_json(folder / "traces.json")
    saved_usage = read_json(folder / "usage.json")

    if manifest.get("stage") not in ("extraction", "finished"):
        raise ValueError("This script resumes extraction only")

    if manifest.get("status") == "completed":
        print("Run already completed. No API calls made.")
        return

    validate_records = runpy.run_path(
        str(ROOT / "scripts/build_entity_index.py")
    )["validate_records"]

    prompt = (
        ROOT / "prompts/ensimem/index_extraction.md"
    ).read_text(encoding="utf-8")

    prompt += """
Additional source-handling rules:
- Preserve the identity of each source turn.
- Captions and image search queries are auxiliary textual observations,
  not verified user statements. Preserve their uncertainty and origin.
- Image URLs are opaque provenance. Do not fetch them or infer their content.
- Ignore operational metadata such as re-download when extracting facts.
- Do not invent a timezone or a precise event time.
"""

    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    if prompt_hash != manifest["extraction_prompt_sha256"]:
        raise ValueError("Extraction prompt changed; cannot silently resume")

    load_dotenv(ROOT / ".env", override=True)

    config = runpy.run_path(
        str(ROOT / "services/llm-gateway/app/config.py")
    )["GatewayConfig"].from_env()

    if config.model != manifest["requested_model"]:
        raise ValueError("Requested model changed since the original run")

    complete_chat = runpy.run_path(
        str(ROOT / "services/llm-gateway/app/providers/openai_compatible.py")
    )["complete_chat"]

    backup = folder.with_name(
        folder.name + "-backup-" + uuid.uuid4().hex[:8]
    )
    shutil.copytree(folder, backup)
    print("Backup:", backup, flush=True)

    ledger = TokenLedger()
    for item in saved_usage:
        ledger.record(TokenUsage(**item))

    cached = {}
    for trace in traces:
        if trace["stage"] == "entity_property_extraction":
            cached[trace["source_id"]] = trace

    records = []
    completed_ids = []
    normalizations = []
    run_id = manifest["experiment_id"]

    manifest["status"] = "incomplete"
    manifest["stage"] = "extraction"
    manifest["normalization_policy"] = (
        "group -> other; original type retained; raw response unchanged"
    )

    def checkpoint():
        manifest["record_count"] = len(records)
        manifest["extracted_episode_count"] = len(completed_ids)

        for filename, value in {
            "manifest.json": manifest,
            "records.json": records,
            "traces.json": traces,
            "usage.json": ledger.export_records(),
            "token_summary.json": ledger.summary(run_id),
            "normalizations.json": normalizations,
            "extracted_episode_ids.json": completed_ids,
        }.items():
            save_json(folder / filename, value)

    try:
        for position, episode in enumerate(episodes, 1):
            episode_id = episode["episode_id"]
            trace = cached.get(episode_id)

            if trace is None:
                print(
                    f"API extraction {position}/{len(episodes)}: {episode_id}",
                    flush=True,
                )

                result = complete_chat(
                    config=config,
                    ledger=ledger,
                    messages=[
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    key: value
                                    for key, value in episode.items()
                                    if key != "theme"
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    experiment_id=run_id,
                    approach="ensimem",
                    phase="build",
                    stage="entity_property_extraction",
                    provider="openai",
                    max_completion_tokens=8000,
                )

                trace = {
                    "source_id": episode_id,
                    "stage": "entity_property_extraction",
                    "call_id": result["call_id"],
                    "latency_seconds": result["request_latency_seconds"],
                    "response": result["response"],
                }
                traces.append(trace)

            else:
                print(
                    f"Reusing saved response {position}/{len(episodes)}",
                    flush=True,
                )

            choice = trace["response"]["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError(
                    f"Incomplete response for {episode_id}"
                )

            parsed = parse_model_json(choice["message"]["content"])
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object")

            items = parsed.get("records")
            if not isinstance(items, list):
                raise ValueError("Expected a records list")

            for index, item in enumerate(items):
                if isinstance(item, dict) and item.get("entity_type") == "group":
                    item["entity_type_original"] = "group"
                    item["entity_type"] = "other"
                    normalizations.append({
                        "episode_id": episode_id,
                        "record_position": index,
                        "field": "entity_type",
                        "from": "group",
                        "to": "other",
                    })

            extracted = validate_records(items, episode)

            for item in extracted:
                records.append({
                    **item,
                    "record_id": f"{run_id}::r{len(records) + 1:06d}",
                    "episode_id": episode_id,
                    "conversation_id": episode["conversation_id"],
                    "session_id": episode["session_id"],
                    "observed_at": episode["observed_at"],
                    "source_timestamp": episode["source_timestamp"],
                    "timezone_status": episode["timezone_status"],
                })

            completed_ids.append(episode_id)
            checkpoint()

        manifest["status"] = "completed"
        manifest["stage"] = "finished"

    finally:
        checkpoint()
        print("\nStatus:", manifest["status"])
        print("Extracted episodes:", len(completed_ids), "/", len(episodes))
        print("Records:", len(records))
        print("Type normalizations:", len(normalizations))
        print(json.dumps(ledger.summary(run_id), indent=2))


if __name__ == "__main__":
    main()
