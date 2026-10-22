"""Recover failed EnSIMem questions while preserving scored results."""

import copy
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover_answer(query, question, config):
    """Only recover complete, sufficient answers with unambiguous citations."""
    traces_path = query / "traces.json"
    evidence_path = query / "evidence.json"
    if not traces_path.exists() or not evidence_path.exists():
        return False

    traces = [
        t for t in read(traces_path)
        if t.get("stage") == "grounded_answer"
    ]
    if not traces:
        return False

    try:
        choice = traces[-1]["response"]["choices"][0]
        if choice.get("finish_reason") != "stop":
            return False

        answer = copy.deepcopy(
            parse_model_json(choice["message"]["content"])
        )
        if (
            not isinstance(answer.get("answer"), str)
            or answer.get("sufficient") is not True
            or not isinstance(answer.get("citations"), list)
            or not answer["citations"]
        ):
            return False

        supplied = {
            e["episode_id"]: {t["turn_id"] for t in e["turns"]}
            for e in read(evidence_path)
        }
        original = copy.deepcopy(answer["citations"])
        repairs = []

        for citation in answer["citations"]:
            eid = citation["episode_id"]
            tid = citation["turn_id"]

            if tid in supplied.get(eid, set()):
                continue

            matches = [
                candidate for candidate, ids in supplied.items()
                if tid in ids
            ]
            if (
                len(matches) != 1
                or matches[0].rsplit("::", 1)[0]
                != eid.rsplit("::", 1)[0]
            ):
                return False

            repairs.append({
                "turn_id": tid,
                "from_episode": eid,
                "to_episode": matches[0],
            })
            citation["episode_id"] = matches[0]

    except (ValueError, KeyError, TypeError, AttributeError):
        return False

    save(query / "result.json", {
        **answer,
        "status": "completed",
        "experiment_id": query.name,
        "memory_build": config["memory_build"],
        "question": question["question"],
        "question_id": question["question_id"],
        "elapsed_seconds": None,
        "recovery": {
            "original_citations": original,
            "repairs": repairs,
            "answer_text_changed": False,
            "elapsed_seconds_note": "Original wall-clock duration unavailable",
        },
    })
    return True


def merge_usage(old_query, new_query):
    """Retain old attempts alongside new calls, deduplicated by call ID."""
    records = {}
    for folder in (old_query, new_query):
        path = folder / "usage.json"
        if not path.exists():
            continue
        for entry in read(path):
            call_id = entry["call_id"]
            if call_id in records and records[call_id] != entry:
                raise ValueError("Conflicting token usage for " + call_id)
            records[call_id] = entry

    if records:
        save(new_query / "usage.json", list(records.values()))


def main():
    source = ROOT / "outputs/benchmark-0e83a3c3dcf9"
    config = read(source / "config.json")
    runner = ROOT / "scripts/run_memory_experiment.py"
    gateway = ROOT / "services/llm-gateway/app/providers/openai_compatible.py"

    load_dotenv(ROOT / ".env", override=True)
    gateway_config = runpy.run_path(
        str(ROOT / "services/llm-gateway/app/config.py")
    )["GatewayConfig"].from_env()

    # Do not silently mix different models or endpoints.
    checks = {
        "answer_model": gateway_config.model,
        "judge_model": os.environ.get("JUDGE_MODEL", "").strip(),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "").strip(),
        "base_url": gateway_config.base_url,
        "timeout_seconds": gateway_config.timeout_seconds,
        "gateway_sha256": sha(gateway),
    }
    for field, value in checks.items():
        if config.get(field) != value:
            raise SystemExit(
                f"Settings changed: {field}. Stopped before API calls."
            )

    memory = None
    for folder in (ROOT / "outputs").glob("offline-memory-*"):
        path = folder / "manifest.json"
        if "-backup-" in folder.name or not path.exists():
            continue
        manifest = read(path)
        if (
            manifest.get("experiment_id") == config["memory_build"]
            and manifest.get("status") == "completed"
        ):
            memory = folder
            break

    if memory is None:
        raise SystemExit("Original completed memory not found.")

    for name, field in (
        ("records.json", "records_sha256"),
        ("episodes.json", "episodes_sha256"),
    ):
        if sha(memory / name) != config[field]:
            raise SystemExit("Memory changed: " + name)

    helper = ROOT / "scripts/requirement_retrieval_v2.py"
    validate_online_plan = runpy.run_path(str(helper))["validate_plan"]
    memory_signature = hashlib.sha256(json.dumps(
        [read(memory / "records.json"), read(memory / "episodes.json")],
        sort_keys=True, ensure_ascii=False,
    ).encode()).hexdigest()

    identity = hashlib.sha256(
        (
            str(source)
            + sha(runner)
            + sha(helper)
            + sha(Path(__file__))
        ).encode()
    ).hexdigest()[:12]
    destination = ROOT / "outputs" / ("benchmark-recovery-" + identity)
    destination.mkdir(exist_ok=True)

    recovered_config = {
        **config,
        "runner_sha256": sha(runner),
        "retrieval_module_sha256": sha(helper),
        "original_retrieval_module_sha256": config.get(
            "retrieval_module_sha256"
        ),
        "recovery": {
            "source_benchmark": source.name,
            "original_runner_sha256": config["runner_sha256"],
            "recovery_script_sha256": sha(Path(__file__)),
            "scope": (
                "Preserved scored answers plus recovered failed questions; "
                "mixed execution versions, not a fresh uniform benchmark."
            ),
        },
    }
    save(destination / "config.json", recovered_config)

    judge_ledger = TokenLedger()
    for path in (
        source / "judge_usage.json",
        destination / "judge_usage.json",
    ):
        if path.exists():
            for entry in read(path):
                judge_ledger.record(TokenUsage(**entry))

    complete_chat = runpy.run_path(str(gateway))["complete_chat"]
    judge_config = replace(gateway_config, model=config["judge_model"])

    print("Recovery benchmark:", destination, flush=True)

    for question in config["questions"]:
        qid = question["question_id"]
        short = qid.split("::")[-1]
        report_path = destination / (short + ".json")
        old_report = source / (short + ".json")
        old_query = source / (short + "-query")
        new_query = destination / (short + "-query")

        if report_path.exists() and read(report_path).get("status") == "scored":
            print(short, "already scored; skipped.", flush=True)
            continue

        previous = read(old_report) if old_report.exists() else {}

        if previous.get("status") == "scored":
            shutil.copy2(old_report, report_path)
            if old_query.exists():
                shutil.copytree(old_query, new_query, dirs_exist_ok=True)
            save(destination / (short + "-provenance.json"), {
                "method": "preserved scored result",
                "source": source.name,
                "runner_sha256": config["runner_sha256"],
            })
            print(short, "preserved; no API calls.", flush=True)
            continue

        report = {**question, "status": "failed"}

        try:
            if not new_query.exists() and old_query.exists():
                shutil.copytree(old_query, new_query)

            result_path = new_query / "result.json"

            if not result_path.exists() and new_query.exists():
                repaired = recover_answer(new_query, question, config)
                if repaired:
                    print(short, "saved answer recovered.", flush=True)

            if not result_path.exists():
                # Keep the original partial attempt; never overwrite it.
                if new_query.exists():
                    archived = destination / (short + "-original-partial")
                    if archived.exists():
                        raise ValueError(
                            "A recovery attempt also failed. "
                            "Logs preserved; not automatically repeating it."
                        )
                    new_query.rename(archived)

                # Reuse a saved, complete planner response when valid.
                trace_path = old_query / "traces.json"

                if trace_path.exists():
                    for trace in reversed(read(trace_path)):
                        if trace.get("stage") != "requirement_planning_v2":
                            continue

                        try:
                            choice = trace["response"]["choices"][0]
                            if choice.get("finish_reason") != "stop":
                                continue

                            plan = validate_online_plan(parse_model_json(
                                choice["message"]["content"]
                            ))
                        except (
                            ValueError, KeyError, TypeError, AttributeError
                        ):
                            continue

                        save(
                            new_query.with_name(
                                new_query.name + "-resume-plan.json"
                            ),
                            {
                                "question": question["question"],
                                "memory_signature": memory_signature,
                                "plan": plan,
                                "source_call_id": trace.get("call_id"),
                                "source_benchmark": source.name,
                            },
                        )
                        print(
                            short,
                            "saved plan ready; planning skipped.",
                            flush=True,
                        )
                        break

                print(short, "running missing query work...", flush=True)
                completed = subprocess.run(
                    [
                        sys.executable, "-u", str(runner),
                        "--memory", str(memory),
                        "--question", question["question"],
                        "--question-id", qid,
                        "--output-dir", str(new_query),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                (destination / (short + ".log")).write_text(
                    completed.stdout + "\n" + completed.stderr,
                    encoding="utf-8",
                )
                if new_query.exists():
                    merge_usage(old_query, new_query)

                if completed.returncode:
                    raise RuntimeError("Query failed; see " + short + ".log")

            answer = read(result_path)
            if (
                answer.get("status") != "completed"
                or answer.get("question") != question["question"]
                or answer.get("memory_build") != config["memory_build"]
            ):
                raise ValueError("Saved answer identity mismatch")

            report.update({
                "generated_answer": answer["answer"],
                "citations": answer["citations"],
                "query_elapsed_seconds": answer.get("elapsed_seconds"),
                "recovery_source": source.name,
            })

            judge_path = destination / (short + "-judge.json")
            if judge_path.exists():
                body = read(judge_path)
            else:
                print(short, "judging saved answer...", flush=True)
                response = complete_chat(
                    config=judge_config,
                    ledger=judge_ledger,
                    messages=[
                        {"role": "system", "content": config["judge_prompt"]},
                        {
                            "role": "user",
                            "content": json.dumps({
                                "question": question["question"],
                                "reference_answer": question["gold_answer"],
                                "generated_answer": answer["answer"],
                            }, ensure_ascii=False),
                        },
                    ],
                    experiment_id=destination.name,
                    approach="ensimem",
                    phase="evaluation",
                    stage="answer_judging",
                    provider="openai",
                    query_id=qid,
                    max_completion_tokens=600,
                )
                body = response["response"]
                save(judge_path, body)

            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("Incomplete judge response")

            verdict = parse_model_json(choice["message"]["content"])
            if verdict.get("label") not in ("CORRECT", "WRONG"):
                raise ValueError("Invalid judge verdict")

            report.update({
                "status": "scored",
                "label": verdict["label"],
                "judge_reason": verdict.get("reason", ""),
            })
            print(short, report["label"], flush=True)

        except Exception as error:
            report["error"] = str(error)
            print(short, "FAILED:", error, flush=True)
        finally:
            save(report_path, report)
            save(
                destination / "judge_usage.json",
                judge_ledger.export_records(),
            )

    reports = [
        read(destination / (q["question_id"].split("::")[-1] + ".json"))
        for q in config["questions"]
    ]
    scored = [r for r in reports if r.get("status") == "scored"]
    correct = sum(r.get("label") == "CORRECT" for r in scored)

    summary = {
        "selected": len(reports),
        "scored": len(scored),
        "failed": len(reports) - len(scored),
        "correct": correct,
        "accuracy_on_scored_percent": (
            100 * correct / len(scored) if scored else None
        ),
        "provenance": recovered_config["recovery"],
    }
    save(destination / "recovery_summary.json", summary)
    print("\nRECOVERY SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nDashboard run:", destination.name)


if __name__ == "__main__":
    main()
