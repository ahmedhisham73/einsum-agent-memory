"""Repair output parsing and error visibility without changing saved results."""

import ast
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
backup = ROOT / "outputs" / (
    "runtime-backup-"
    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
)
backup.mkdir(parents=True, exist_ok=True)


def write_code(path, text):
    ast.parse(text)
    if path.exists():
        relative = path.relative_to(ROOT)
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# Accept a complete JSON response, optionally wrapped in one Markdown fence.
parser_path = ROOT / "packages/contracts/json_output.py"
write_code(parser_path, '''\
"""Parse model JSON without accepting unrelated surrounding prose."""

import json
import re


def parse_model_json(content):
    if not isinstance(content, str):
        raise ValueError("Model content must be text")

    value = content.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\\s*\\n?(.*?)\\s*```",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        value = fenced.group(1).strip()

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a model JSON object")
    return parsed
''')

names = {
    "build_episodes.py",
    "build_entity_index.py",
    "build_offline_memory.py",
    "resume_offline_memory.py",
    "run_memory_experiment.py",
    "run_agentic_rag.py",
    "benchmark_memory.py",
    "benchmark_agentic_rag.py",
    "benchmark_all_ensimem.py",
    "benchmark_all_agentic.py",
    "benchmark_final_ensimem.py",
    "benchmark_final_agentic.py",
}

for name in sorted(names):
    path = ROOT / "scripts" / name
    if not path.exists():
        continue

    original = path.read_text()
    updated = original.replace(
        'json.loads(choice["message"]["content"])',
        'parse_model_json(choice["message"]["content"])',
    )

    # The resume script parses a named content variable.
    if name == "resume_offline_memory.py":
        updated = updated.replace(
            "json.loads(content)", "parse_model_json(content)"
        )

    if updated != original:
        import_line = (
            "from packages.contracts.json_output import parse_model_json"
        )
        if import_line not in updated:
            anchor = "sys.path.insert(0, str(ROOT))"
            if anchor not in updated:
                raise ValueError(f"Import anchor missing: {name}")
            updated = updated.replace(
                anchor, anchor + "\n\n" + import_line, 1
            )
        write_code(path, updated)
        print("JSON parser repaired:", name)


# Preserve unknown entity semantics via original type; keep reported as reported.
path = ROOT / "scripts/build_entity_index.py"
text = path.read_text()

if "_strict_validate_records = validate_records" not in text:
    anchor = "def main():"
    if anchor not in text:
        raise ValueError("Index validator insertion point missing")

    addition = '''\
# "reported" retains source-reporting semantics; it is not mapped to certainty.
MODALITIES.add("reported")
_strict_validate_records = validate_records


def validate_records(records, episode):
    if isinstance(records, list):
        for record in records:
            if (
                isinstance(record, dict)
                and record.get("entity_type") in ("animal", "group")
            ):
                record.setdefault(
                    "entity_type_original", record["entity_type"]
                )
                record["entity_type"] = "other"
    return _strict_validate_records(records, episode)


'''
    write_code(path, text.replace(anchor, addition + anchor, 1))
    print("Index schema compatibility repaired.")


# Reveal HTTP/network/validation causes without printing credentials.
path = ROOT / "services/llm-gateway/app/providers/openai_compatible.py"
text = path.read_text()

old_message = (
    'f"LLM request failed; call_id={call_id}. Usage is unknown."'
)
new_message = (
    'f"LLM request failed; call_id={call_id}; '
    '{_safe_failure(exc, config.api_key)}"'
)

if old_message in text:
    anchor = "def complete_chat("
    if anchor not in text:
        raise ValueError("Gateway function not found")

    helper = '''\
def _safe_failure(exc, api_key):
    status = getattr(exc, "code", None)
    details = [f"kind={type(exc).__name__}"]

    if status is not None:
        details.append(f"HTTP={status}")
        try:
            body = json.loads(exc.read().decode("utf-8"))
            error = body.get("error", {})
            if isinstance(error, dict):
                for field in ("type", "code"):
                    if error.get(field) is not None:
                        details.append(f"{field}={error[field]}")
        except Exception:
            pass
    else:
        details.append(str(exc)[:240])

    message = "; ".join(details)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message


'''
    text = text.replace(anchor, helper + anchor, 1)

    # This is the broad exception handler in complete_chat.
    text = text.replace(
        "    except Exception:\n"
        "        # A failed request may still have consumed tokens.",
        "    except Exception as exc:\n"
        "        # A failed request may still have consumed tokens.",
        1,
    )

    if "    except Exception as exc:" not in text:
        raise ValueError("Gateway exception handler not found")

    text = text.replace(old_message, new_message, 1)
    write_code(path, text)
    print("Gateway error diagnostics repaired.")

print("\nDONE. Code backup:", backup)
print("Saved experiment data was not modified.")
