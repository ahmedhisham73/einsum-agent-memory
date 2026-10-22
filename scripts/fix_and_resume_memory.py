"""Repair compatibility, validate saved responses, then resume missing work."""

import ast
import json
import os
import runpy
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    recovery = ROOT / "scripts/recover_offline_memories.py"
    validator = ROOT / "scripts/build_entity_index.py"

    for path in (
        recovery,
        validator,
        ROOT / "packages/contracts/json_output.py",
    ):
        if not path.exists():
            raise SystemExit(f"Required file missing: {path}")

    backup = ROOT / "outputs" / (
        "compatibility-backup-" + uuid.uuid4().hex[:12]
    )
    backup.mkdir(parents=True)

    def update(path, content):
        ast.parse(content)
        if path.read_text() == content:
            return
        destination = backup / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        path.write_text(content, encoding="utf-8")

    # Normalize index representations without discarding original values.
    text = validator.read_text()
    start = text.find('# "reported" retains source-reporting semantics')
    end = text.find("def main():", start)

    if start < 0 or end < 0:
        raise ValueError("Expected validator wrapper not found")

    wrapper = '''\
# "reported" retains source-reporting semantics.
MODALITIES.add("reported")
_strict_validate_records = validate_records


def validate_records(records, episode):
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue

            entity_type = record.get("entity_type")
            if (
                isinstance(entity_type, str)
                and entity_type.strip()
                and entity_type not in ENTITY_TYPES
            ):
                record.setdefault("entity_type_original", entity_type)
                record["entity_type"] = "other"

            for field in ("value", "valid_time"):
                if field not in record:
                    continue

                original = record[field]
                if isinstance(original, str):
                    continue

                record.setdefault(field + "_original", original)
                record[field] = (
                    "" if original is None
                    else json.dumps(
                        original, ensure_ascii=False, sort_keys=True
                    )
                )

    return _strict_validate_records(records, episode)


'''
    update(validator, text[:start] + wrapper + text[end:])
    print("Record compatibility ready; original values retained.")

    # Bounded retries for observed transient DNS failures.
    text = recovery.read_text()
    text = text.replace(
        "for attempt in (1, 2):",
        "for attempt in (1, 2, 3, 4):",
    ).replace(
        "if attempt == 2 or not transient:",
        "if attempt == 4 or not transient:",
    )

    if '"nodename nor servname"' not in text:
        anchor = '"ConnectionResetError",'
        if anchor not in text:
            raise ValueError("Recovery retry configuration not found")
        text = text.replace(
            anchor,
            anchor + '\n'
            '                                "nodename nor servname",\n'
            '                                "Name or service not known",\n'
            '                                "Temporary failure in name resolution",',
            1,
        )

    text = text.replace(
        "time.sleep(3)",
        "time.sleep(min(3 * 2 ** (attempt - 1), 24))",
    )
    text = text.replace(
        "    results = {}",
        '    state_path = ROOT / "outputs/offline-recovery-v11.json"\n'
        '    results = read(state_path) if state_path.exists() else {}',
    )
    update(recovery, text)
    print("Recovery retries ready.")
    print("Code backup:", backup)

    from packages.contracts.dataset import Session, Turn
    from packages.contracts.json_output import parse_model_json

    validate_segments = runpy.run_path(
        str(ROOT / "scripts/build_episodes.py")
    )["validate_segments"]
    validate_records = runpy.run_path(str(validator))["validate_records"]

    errors = []
    checked = 0

    # Preflight cached data without making model calls.
    for folder in (ROOT / "outputs").glob("offline-memory-*"):
        if "-backup-" in folder.name:
            continue

        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read(manifest_path)

        if (
            manifest.get("status") == "completed"
            or manifest.get("source_kind") != "locomo_benchmark"
        ):
            continue

        source = read(folder / "source_conversation.json")
        cached = {}

        for trace in read(folder / "traces.json"):
            source_id = trace.get("source_id", trace.get("episode_id"))
            if source_id:
                cached[(trace["stage"], source_id)] = trace

        episode_map = {}

        for item in source["sessions"]:
            session_id = item["session_id"]
            trace = cached.get(("episode_partitioning", session_id))
            if trace is None:
                continue

            try:
                choice = trace["response"]["choices"][0]
                if choice.get("finish_reason") != "stop":
                    continue

                parsed = parse_model_json(choice["message"]["content"])
                session = Session(
                    session_id=session_id,
                    observed_at=item["observed_at"],
                    source_timestamp=item.get("source_timestamp", ""),
                    timezone_status=item.get("timezone_status", "specified"),
                    turns=tuple(Turn(**turn) for turn in item["turns"]),
                )

                segments = validate_segments(
                    parsed.get("segments"), session
                )
                for number, (first, last, theme) in enumerate(segments, 1):
                    episode_id = (
                        f'{source["conversation_id"]}::'
                        f'{session_id}::episode-{number:03d}'
                    )
                    episode_map[episode_id] = {
                        "episode_id": episode_id,
                        "turns": item["turns"][first:last + 1],
                    }
                checked += 1

            except Exception as error:
                errors.append(
                    f"{folder.name} / {session_id}: {error}"
                )

        for (stage, episode_id), trace in cached.items():
            if stage != "entity_property_extraction":
                continue
            try:
                choice = trace["response"]["choices"][0]
                if choice.get("finish_reason") != "stop":
                    continue
                if episode_id not in episode_map:
                    raise ValueError("No validated parent episode")

                parsed = parse_model_json(choice["message"]["content"])
                validate_records(
                    parsed.get("records"), episode_map[episode_id]
                )
                checked += 1

            except Exception as error:
                errors.append(
                    f"{folder.name} / {episode_id}: {error}"
                )

    print("Saved responses checked:", checked)

    if errors:
        report = ROOT / "outputs/cached_validation_errors.txt"
        report.write_text("\n".join(errors), encoding="utf-8")
        print("\n".join(errors[:12]))
        raise SystemExit(
            f"Stopped before new API calls. Details: {report}"
        )

    print("PASS: cached responses passed structural validation.")

    # Check endpoint/proxy DNS before starting paid work.
    load_dotenv(ROOT / ".env", override=True)
    api_host = urllib.parse.urlsplit(
        os.environ.get("LLM_BASE_URL", "")
    ).hostname
    if not api_host:
        raise ValueError("Invalid LLM_BASE_URL")

    proxy_url = urllib.request.getproxies().get("https")
    target = api_host
    if proxy_url and not urllib.request.proxy_bypass(api_host):
        target = urllib.parse.urlsplit(
            proxy_url if "://" in proxy_url else "//" + proxy_url
        ).hostname

    if not target:
        raise ValueError("Invalid proxy hostname")

    for attempt in range(3):
        try:
            socket.getaddrinfo(target, None)
            print("PASS: hostname resolved:", target)
            break
        except socket.gaierror:
            if attempt == 2:
                raise SystemExit(
                    "DNS unavailable. Restore connectivity and rerun "
                    "this script. No model API calls were made."
                )
            time.sleep(2)

    print("\nResuming missing offline work...", flush=True)
    subprocess.run(
        [sys.executable, "-u", str(recovery)],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
