"""Extract provenance-linked index records from a completed episode build."""

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

ENTITY_TYPES = {
    "person", "organization", "place", "object",
    "event", "activity", "concept", "other",
}
MODALITIES = {
    "observed", "planned", "desired",
    "hypothetical", "negated", "uncertain",
}


def validate_records(records, episode):
    if not isinstance(records, list):
        raise ValueError("Expected a records list")

    allowed_ids = {turn["turn_id"] for turn in episode["turns"]}

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be an object")

        for key in ("entity", "entity_type", "property", "modality"):
            value = record.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing or invalid {key}")

        for key in ("value", "valid_time"):
            if not isinstance(record.get(key), str):
                raise ValueError(f"{key} must be a string")

        if record["entity_type"] not in ENTITY_TYPES:
            raise ValueError("Unsupported entity_type")
        if record["modality"] not in MODALITIES:
            raise ValueError("Unsupported modality")

        evidence = record.get("evidence_turn_ids")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Every record requires evidence turn IDs")
        if not all(isinstance(item, str) for item in evidence):
            raise ValueError("Evidence IDs must be strings")
        if not set(evidence).issubset(allowed_ids):
            raise ValueError("Evidence references a turn outside the episode")

    return records


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/build_entity_index.py EPISODE_BUILD_FOLDER"
        )

    load_dotenv(ROOT / ".env", override=True)

    source_dir = Path(sys.argv[1]).resolve()
    manifest_path = source_dir / "manifest.json"
    episodes_path = source_dir / "episodes.json"

    parent_manifest = json.loads(manifest_path.read_text())
    if parent_manifest.get("status") != "completed":
        raise ValueError("Episode build must be completed")

    episodes = json.loads(episodes_path.read_text())
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("No episodes to index")

    config = runpy.run_path(
        str(ROOT / "services/llm-gateway/app/config.py")
    )["GatewayConfig"].from_env()

    complete_chat = runpy.run_path(
        str(ROOT / "services/llm-gateway/app/providers/openai_compatible.py")
    )["complete_chat"]

    prompt_path = ROOT / "prompts/ensimem/index_extraction.md"
    prompt = prompt_path.read_text(encoding="utf-8")

    experiment_id = "entity-index-" + uuid.uuid4().hex[:12]
    output_dir = ROOT / "outputs" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=False)

    ledger = TokenLedger()
    records = []
    traces = []
    status = "incomplete"

    try:
        for episode in episodes:
            print("Indexing:", episode["episode_id"], flush=True)

            # Deliberately exclude the model-generated theme from evidence.
            payload = {
                "episode_id": episode["episode_id"],
                "observed_at": episode["observed_at"],
                "turns": episode["turns"],
            }

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
                experiment_id=experiment_id,
                approach="ensimem",
                phase="build",
                stage="entity_property_extraction",
                provider="openai",
                max_completion_tokens=3000,
            )

            body = result["response"]
            traces.append({
                "episode_id": episode["episode_id"],
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

            extracted = validate_records(parsed.get("records"), episode)

            for item in extracted:
                records.append({
                    **item,
                    "record_id": f"{experiment_id}::r{len(records) + 1:04d}",
                    "episode_id": episode["episode_id"],
                    "conversation_id": episode["conversation_id"],
                    "session_id": episode["session_id"],
                    "observed_at": episode["observed_at"],
                })

        status = "completed"

    finally:
        artifacts = {
            "manifest.json": {
                "experiment_id": experiment_id,
                "status": status,
                "parent_episode_build": parent_manifest["experiment_id"],
                "parent_episode_folder": str(source_dir),
                "dataset_sha256": parent_manifest["dataset_sha256"],
                "episodes_sha256": hashlib.sha256(
                    episodes_path.read_bytes()
                ).hexdigest(),
                "prompt_sha256": hashlib.sha256(
                    prompt_path.read_bytes()
                ).hexdigest(),
                "requested_model": config.model,
                "record_count": len(records),
            },
            "records.json": records,
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

    print("\nValidated index records:", len(records))


if __name__ == "__main__":
    main()
