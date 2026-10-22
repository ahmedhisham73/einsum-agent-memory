"""Persistent embeddings + Qdrant + one grounded memory query."""

import argparse
import hashlib
import json
import math
import os
import re
import runpy
import sys
import time
import urllib.request
import uuid
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.contracts.json_output import parse_model_json

from packages.contracts.usage import TokenUsage
from packages.telemetry.token_ledger import TokenLedger


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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def record_text(record):
    fields = ("entity", "entity_type", "property", "value",
              "modality", "valid_time")
    return "\n".join(
        f"{field}: {record.get(field, '')}" for field in fields
    )


def tokens(text):
    return set(re.findall(r"\w+", text.casefold()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--memory", default="outputs/offline-memory-a5e48166f56e"
    )
    parser.add_argument(
        "--question", default="What did Caroline research?"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--question-id", default="q-001")
    args = parser.parse_args()

    if not args.question.strip():
        raise ValueError("Question must not be empty")

    load_dotenv(ROOT / ".env", override=True)
    config = runpy.run_path(
        str(ROOT / "services/llm-gateway/app/config.py")
    )["GatewayConfig"].from_env()

    provider = runpy.run_path(
        str(ROOT / "services/llm-gateway/app/providers/openai_compatible.py")
    )
    complete_chat = provider["complete_chat"]
    opener = urllib.request.build_opener(provider["NoRedirect"]())

    embedding_model = os.environ.get("EMBEDDING_MODEL", "").strip()
    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL is missing from .env")

    memory = Path(args.memory).resolve()
    parent = read(memory / "manifest.json")
    if parent["status"] != "completed":
        raise ValueError("Offline memory must be completed")

    records = read(memory / "records.json")
    episodes = read(memory / "episodes.json")
    episode_map = {item["episode_id"]: item for item in episodes}
    if not records:
        raise ValueError("Memory contains no records")

    texts = [record_text(record) for record in records]
    fingerprint = digest({
        "records": records,
        "texts": texts,
        "embedding_model": embedding_model,
        "base_url": config.base_url,
    })

    dense_dir = memory / ("dense-" + fingerprint[:12])
    dense_dir.mkdir(exist_ok=True)
    cache_path = dense_dir / "cache.json"

    if cache_path.exists():
        state = read(cache_path)
        if state["fingerprint"] != fingerprint:
            raise ValueError("Embedding cache fingerprint mismatch")
    else:
        state = {
            "fingerprint": fingerprint,
            "experiment_id": "embedding-build-" + uuid.uuid4().hex[:12],
            "vectors": [],
            "usage": [],
        }

    build_ledger = TokenLedger()
    for item in state["usage"]:
        build_ledger.record(TokenUsage(**item))

    def embed(inputs, ledger, experiment_id, phase, stage, query_id=None):
        call_id = str(uuid.uuid4())
        metadata = dict(
            call_id=call_id,
            experiment_id=experiment_id,
            approach="ensimem",
            phase=phase,
            stage=stage,
            provider="openai",
            model=embedding_model,
            query_id=query_id,
            usage_kind="embedding",
        )
        TokenUsage(**metadata)
        usage_record = None

        request = urllib.request.Request(
            config.base_url + "/embeddings",
            data=json.dumps({
                "model": embedding_model,
                "input": inputs,
                "encoding_format": "float",
            }).encode(),
            headers={
                "Authorization": "Bearer " + config.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with opener.open(
                request, timeout=config.timeout_seconds
            ) as response:
                body = json.loads(response.read().decode())

            if not isinstance(body, dict) or body.get("error"):
                raise ValueError("Invalid embedding response")

            reported = body.get("usage") or {}
            usage_record = TokenUsage(
                **{**metadata, "model": body.get("model") or embedding_model},
                input_tokens=reported.get("prompt_tokens"),
                output_tokens=0,
            )

            total = reported.get("total_tokens")
            if total is not None:
                if type(total) is not int or total < 0:
                    raise ValueError("Invalid embedding total")
                if (
                    usage_record.input_tokens is not None
                    and total != usage_record.input_tokens
                ):
                    raise ValueError("Embedding token total mismatch")

            items = body["data"]
            indices = [item["index"] for item in items]
            if any(type(index) is not int for index in indices):
                raise ValueError("Invalid vector index")
            if sorted(indices) != list(range(len(inputs))):
                raise ValueError("Missing or duplicated vectors")

            vectors = [
                item["embedding"]
                for item in sorted(items, key=lambda item: item["index"])
            ]
            dimension = len(vectors[0])
            if dimension == 0:
                raise ValueError("Empty embedding")

            for vector in vectors:
                if len(vector) != dimension or not all(
                    type(value) in (int, float) and math.isfinite(value)
                    for value in vector
                ):
                    raise ValueError("Invalid embedding vector")

            return vectors
        finally:
            ledger.record(usage_record or TokenUsage(**metadata))

    # Completed embedding batches are cached and never requested again.
    try:
        while len(state["vectors"]) < len(records):
            start = len(state["vectors"])
            batch = texts[start:start + 32]
            print(
                f"Embedding records {start + 1}-"
                f"{start + len(batch)}/{len(records)}",
                flush=True,
            )
            vectors = embed(
                batch, build_ledger, state["experiment_id"],
                "build", "index_embedding",
            )
            if state["vectors"]:
                if len(vectors[0]) != len(state["vectors"][0]):
                    raise ValueError("Embedding dimensions changed")

            state["vectors"].extend(vectors)
            state["usage"] = build_ledger.export_records()
            save(cache_path, state)
    finally:
        state["usage"] = build_ledger.export_records()
        save(cache_path, state)

    print("Embeddings ready:", len(state["vectors"]), flush=True)

    client = QdrantClient(path=str(dense_dir / "qdrant"))
    collection = "ensimem_records"

    query_run = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else ROOT / "outputs" / ("query-" + uuid.uuid4().hex[:12])
    )
    query_run.mkdir()
    query_id = args.question_id
    query_ledger = TokenLedger()
    query_traces = []
    started = time.perf_counter()

    try:
        names = {c.name for c in client.get_collections().collections}
        if collection not in names:
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=len(state["vectors"][0]),
                    distance=models.Distance.COSINE,
                ),
            )

        # Stable IDs make repeated upserts idempotent.
        for start in range(0, len(records), 64):
            client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=index,
                        vector=state["vectors"][index],
                        payload={
                            "record_id": records[index]["record_id"],
                            "episode_id": records[index]["episode_id"],
                        },
                    )
                    for index in range(start, min(start + 64, len(records)))
                ],
                wait=True,
            )

        count = client.count(collection_name=collection, exact=True).count
        if count != len(records):
            raise ValueError("Qdrant record count mismatch")
        print("Qdrant records:", count, flush=True)

        def chat(system, payload, stage):
            result = complete_chat(
                config=config,
                ledger=query_ledger,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                experiment_id=query_run.name,
                approach="ensimem",
                phase="query",
                stage=stage,
                provider="openai",
                query_id=query_id,
                max_completion_tokens=1800,
            )
            query_traces.append({
                "stage": stage,
                "call_id": result["call_id"],
                "response": result["response"],
                "latency_seconds": result["request_latency_seconds"],
            })
            save(query_run / "traces.json", query_traces)

            choice = result["response"]["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Incomplete model response")

            parsed = parse_model_json(choice["message"]["content"])
            if not isinstance(parsed, dict):
                raise ValueError("Expected JSON object")
            return parsed


        # Robust structured model responses v1
        _original_chat = chat

        def validate_response(parsed, payload, stage):
            if stage == "requirement_planning":
                requirements = parsed.get("requirements")
                if (
                    parsed.get("scope") not in ("point", "all_matching")
                    or not isinstance(requirements, list)
                    or not 1 <= len(requirements) <= 8
                ):
                    raise ValueError("Invalid retrieval plan")

                for requirement in requirements:
                    if not isinstance(requirement, dict):
                        raise ValueError("Invalid requirement")
                    for field in ("entity", "property", "search"):
                        if not isinstance(requirement.get(field), str):
                            raise ValueError("Invalid requirement field")
                    if not requirement["search"].strip():
                        raise ValueError("Empty requirement search")

            elif stage == "grounded_answer":
                if (
                    not isinstance(parsed.get("answer"), str)
                    or type(parsed.get("sufficient")) is not bool
                    or not isinstance(parsed.get("citations"), list)
                ):
                    raise ValueError("Invalid answer schema")

                supplied = {
                    episode["episode_id"]: {
                        turn["turn_id"] for turn in episode["turns"]
                    }
                    for episode in payload["episodes"]
                }
                original = json.loads(json.dumps(parsed["citations"]))
                repairs = []

                for citation in parsed["citations"]:
                    if not isinstance(citation, dict):
                        raise ValueError("Invalid citation schema")

                    episode_id = citation.get("episode_id")
                    turn_id = citation.get("turn_id")

                    if (
                        not isinstance(episode_id, str)
                        or not isinstance(turn_id, str)
                    ):
                        raise ValueError("Invalid citation IDs")

                    if turn_id in supplied.get(episode_id, set()):
                        continue

                    matches = [
                        eid for eid, turns in supplied.items()
                        if turn_id in turns
                    ]

                    if (
                        len(matches) != 1
                        or matches[0].rsplit("::", 1)[0]
                        != episode_id.rsplit("::", 1)[0]
                    ):
                        raise ValueError("Unresolved citation")

                    repairs.append({
                        "turn_id": turn_id,
                        "original_episode_id": episode_id,
                        "corrected_episode_id": matches[0],
                    })
                    citation["episode_id"] = matches[0]

                if parsed["sufficient"] and not parsed["citations"]:
                    raise ValueError("Supported answer requires citations")

                if repairs:
                    parsed["citation_recovery"] = {
                        "original_citations": original,
                        "repairs": repairs,
                        "method": "unique exact turn within same session",
                    }

            return parsed

        def chat(system, payload, stage):
            last_error = None

            for attempt in range(1, 4):
                instruction = system
                if attempt > 1:
                    instruction += (
                        "\nYour previous response failed structural validation. "
                        "Return one complete, concise JSON object. "
                        "Copy citation episode_id and turn_id exactly from "
                        "the same supplied episode. Do not invent IDs. "
                        "Keep requirement fields short."
                    )

                try:
                    parsed = _original_chat(instruction, payload, stage)
                    return validate_response(parsed, payload, stage)
                except (ValueError, KeyError, TypeError) as error:
                    last_error = error
                    print(
                        f"{stage}: invalid response "
                        f"({attempt}/3): {error}",
                        flush=True,
                    )

            raise ValueError(
                f"{stage}: failed structural validation after 3 attempts"
            ) from last_error


        # Requirement coverage retrieval v2
        retrieval_module = runpy.run_path(
            str(ROOT / "scripts/requirement_retrieval_v2.py")
        )
        final = retrieval_module["retrieve"](
            question=args.question,
            records=records,
            episodes=episodes,
            client=client,
            collection=collection,
            chat=locals().get("_original_chat", chat),
            embed_requirements=lambda searches: embed(
                searches,
                query_ledger,
                query_run.name,
                "query",
                "requirement_embedding",
                query_id,
            ),
            output=query_run,
            save=save,
        )

        result = {
            "status": "completed",
            "experiment_id": query_run.name,
            "memory_build": parent["experiment_id"],
            "embedding_fingerprint": fingerprint,
            "question": args.question,
            **final,
            "elapsed_seconds": time.perf_counter() - started,
            "implementation": {
                "retrieval_version": "requirement-coverage-v2",
                "structured_matching": "semantic index compatibility",
                "all_matching": "full index scan; matching source episodes only",
                "coverage": "per-requirement source citations; model assessment",
            },
        }
        save(query_run / "result.json", result)
        print("\nANSWER:", result["answer"])
        print("CITATIONS:", json.dumps(result["citations"], indent=2))
        print("Evidence sufficient:", result["sufficient"])
        print("Artifacts:", query_run)

    finally:
        save(query_run / "usage.json", query_ledger.export_records())
        save(
            query_run / "token_summary.json",
            query_ledger.summary(query_run.name),
        )
        save(
            dense_dir / "token_summary.json",
            build_ledger.summary(state["experiment_id"]),
        )
        client.close()

    print("\nEMBEDDING BUILD USAGE:")
    print(json.dumps(
        build_ledger.summary(state["experiment_id"]), indent=2
    ))
    print("\nQUERY USAGE:")
    print(json.dumps(query_ledger.summary(query_run.name), indent=2))


if __name__ == "__main__":
    main()
