"""Run the existing comparison protocol across all LoCoMo conversations."""

import csv
import hashlib
import html
import json
import os
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
JOB = OUTPUTS / "full-locomo-v1"
JOB.mkdir(parents=True, exist_ok=True)


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


def execute(command, log_path):
    """Stream useful progress and retain complete command output."""
    found = {}
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()

            for prefix in ("Output:", "Benchmark:"):
                if line.startswith(prefix):
                    found[prefix] = line.split(":", 1)[1].strip()

            if line.startswith((
                "Partitioning", "Extracting", "Reusing",
                "[", "  CORRECT", "  WRONG", "  FAILED",
                "Status:", "Benchmark:",
            )):
                print(line, end="", flush=True)

        code = process.wait()

    return code, found


def create_cached_benchmark(original_name, target_name):
    """Reuse scored questions only when execution settings match."""
    source = (ROOT / "scripts" / original_name).read_text()

    anchor = "    for number, item in enumerate(selected, 1):"
    if anchor not in source:
        raise ValueError(f"Benchmark loop not found: {original_name}")

    cache_code = textwrap.dedent("""
        # Import compatible scored results without making new API calls.
        reused = {}
        compatibility_fields = (
            "memory_build", "raw_dataset_sha256", "runner_sha256",
            "gateway_sha256", "answer_model", "judge_model",
            "embedding_model", "judge_prompt", "base_url",
            "timeout_seconds",
        )

        for previous in sorted((ROOT / "outputs").glob("benchmark-*")):
            if previous == folder or not (previous / "config.json").exists():
                continue

            previous_config = read(previous / "config.json")
            if any(
                previous_config.get(key) != settings.get(key)
                for key in compatibility_fields
            ):
                continue

            for selected_item in selected:
                qid = selected_item["question_id"]
                short_id = qid.split("::")[-1]
                target_report = folder / f"{short_id}.json"
                previous_report = previous / f"{short_id}.json"
                previous_query = previous / f"{short_id}-query"

                if target_report.exists() or not previous_report.exists():
                    continue
                if not (previous_query / "result.json").exists():
                    continue
                if not (previous_query / "usage.json").exists():
                    continue

                cached_report = read(previous_report)
                if (
                    cached_report.get("status") != "scored"
                    or cached_report.get("question_id") != qid
                    or cached_report.get("question") != selected_item["question"]
                    or cached_report.get("gold_answer")
                       != selected_item["gold_answer"]
                ):
                    continue

                shutil.copy2(previous_report, target_report)
                shutil.copytree(
                    previous_query,
                    folder / f"{short_id}-query",
                )

                judge_trace = previous / f"{short_id}-judge.json"
                if judge_trace.exists():
                    shutil.copy2(judge_trace, folder / judge_trace.name)

                previous_usage = previous / "judge_usage.json"
                if previous_usage.exists():
                    for entry in read(previous_usage):
                        if entry.get("query_id") == qid:
                            judge_ledger.record(TokenUsage(**entry))

                reused[qid] = str(previous)

        if reused:
            save(folder / "cache_provenance.json", reused)
            save(judge_usage_path, judge_ledger.export_records())
            print("Reused scored questions:", len(reused), flush=True)
    """)

    indented = textwrap.indent(cache_code, "    ")
    source = "import shutil\n" + source.replace(
        anchor, indented + "\n" + anchor, 1
    )
    compile(source, target_name, "exec")
    target = ROOT / "scripts" / target_name
    target.write_text(source, encoding="utf-8")
    return target


def locate_memory(sample_id, raw_hash):
    completed = []
    partial = []

    for folder in OUTPUTS.glob("offline-memory-*"):
        if "-backup-" in folder.name:
            continue
        path = folder / "manifest.json"
        if not path.exists():
            continue
        manifest = read(path)
        if (
            manifest.get("conversation_id") != sample_id
            or manifest.get("raw_dataset_sha256") != raw_hash
        ):
            continue
        if manifest.get("status") == "completed":
            completed.append(folder)
        else:
            partial.append(folder)

    candidates = completed or partial
    return (
        max(candidates, key=lambda p: (p / "manifest.json").stat().st_mtime)
        if candidates else None
    )


def build_memory(sample_id, raw_hash):
    folder = locate_memory(sample_id, raw_hash)
    log = JOB / f"{sample_id}-build.log"

    if folder and read(folder / "manifest.json")["status"] == "completed":
        print("Reusing completed memory:", folder.name, flush=True)
        return folder

    if folder:
        manifest = read(folder / "manifest.json")
        if manifest.get("stage") != "extraction":
            raise RuntimeError(
                f"Partial partitioning build needs inspection: {folder}"
            )
    else:
        code, found = execute(
            [
                sys.executable, "-u",
                str(ROOT / "scripts/build_offline_memory.py"),
                "--sample-id", sample_id,
            ],
            log,
        )
        folder = (
            Path(found["Output:"])
            if "Output:" in found
            else locate_memory(sample_id, raw_hash)
        )
        if not folder:
            raise RuntimeError(f"Build failed; inspect {log}")

        manifest = read(folder / "manifest.json")
        if manifest["status"] == "completed":
            return folder
        if manifest.get("stage") != "extraction":
            raise RuntimeError(f"Build incomplete; inspect {log}")

    # Reuse cached extraction responses and apply the existing group->other policy.
    code, _ = execute(
        [
            sys.executable, "-u",
            str(ROOT / "scripts/resume_offline_memory.py"),
            str(folder),
        ],
        log,
    )
    if code or read(folder / "manifest.json")["status"] != "completed":
        raise RuntimeError(
            f"Extraction incomplete; saved data preserved. Inspect {log}"
        )
    return folder


def benchmark(script, memory, count, log):
    code, found = execute(
        [
            sys.executable, "-u", str(script),
            "--memory", str(memory),
            "--limit", str(count),
        ],
        log,
    )
    if code or "Benchmark:" not in found:
        raise RuntimeError(f"Benchmark failed; inspect {log}")
    folder = Path(found["Benchmark:"])
    read(folder / "summary.json")
    return folder


def export(state, samples):
    rows = []
    totals = {
        approach: {
            "eligible": 0, "scored": 0, "correct": 0,
            "query_generation_tokens_reported": 0,
            "query_embedding_tokens_reported": 0,
            "judge_generation_tokens_reported": 0,
            "usage_complete": True,
        }
        for approach in ("ensimem", "agentic_rag")
    }

    for sample in samples:
        sample_id = sample["sample_id"]
        entry = state.get(sample_id, {})
        eligible = sum(
            q.get("category") in (1, 2, 3, 4) and "answer" in q
            for q in sample["qa"]
        )

        for approach in totals:
            aggregate = totals[approach]
            aggregate["eligible"] += eligible
            folder_text = entry.get(approach)
            if not folder_text:
                continue

            folder = Path(folder_text)
            if not (folder / "summary.json").exists():
                continue

            summary = read(folder / "summary.json")
            aggregate["scored"] += summary["scored"]
            aggregate["correct"] += summary["correct"]

            for group in summary["query_usage"]:
                key = (
                    "query_generation_tokens_reported"
                    if group["usage_kind"] == "generation"
                    else "query_embedding_tokens_reported"
                )
                aggregate[key] += (
                    group["input_tokens_reported"]
                    + group["output_tokens_reported"]
                )
                if group.get("missing_usage_calls", 0):
                    aggregate["usage_complete"] = False

            for group in summary["judge_usage"]:
                aggregate["judge_generation_tokens_reported"] += (
                    group["input_tokens_reported"]
                    + group["output_tokens_reported"]
                )
                if group.get("missing_usage_calls", 0):
                    aggregate["usage_complete"] = False

            config = read(folder / "config.json")
            for question in config["questions"]:
                short_id = question["question_id"].split("::")[-1]
                report_path = folder / f"{short_id}.json"
                report = read(report_path) if report_path.exists() else {}
                rows.append({
                    "conversation": sample_id,
                    "approach": approach,
                    "question_id": question["question_id"],
                    "category": question["category"],
                    "question": question["question"],
                    "gold_answer": json.dumps(
                        question["gold_answer"], ensure_ascii=False
                    ),
                    "generated_answer": report.get("generated_answer", ""),
                    "status": report.get("status", "missing"),
                    "label": report.get("label", ""),
                    "judge_reason": report.get("judge_reason", ""),
                    "error": report.get("error", ""),
                })

    for aggregate in totals.values():
        aggregate["unscored"] = aggregate["eligible"] - aggregate["scored"]
        aggregate["accuracy_on_scored_percent"] = (
            100 * aggregate["correct"] / aggregate["scored"]
            if aggregate["scored"] else None
        )

    complete = (
        len(state) == len(samples)
        and all(
            entry.get("status") == "completed"
            for entry in state.values()
        )
        and all(
            item["unscored"] == 0 for item in totals.values()
        )
    )

    report = {
        "status": "completed" if complete else "incomplete",
        "scope": "All 10 LoCoMo conversations; categories 1-4",
        "evaluation": "Custom LLM judge rubric; prototype implementations",
        "development_questions": (
            "Original 10 included in aggregate; their IDs remain separately "
            "identified in protocol.json. Aggregate is not a pure held-out score."
        ),
        "build_accounting": (
            "Build and embedding-build ledgers retained in each linked build. "
            "Excluded from the query totals below."
        ),
        "results": totals,
        "conversations": state,
    }

    save(JOB / "summary.json", report)

    fields = [
        "conversation", "approach", "question_id", "category", "question",
        "gold_answer", "generated_answer", "status", "label",
        "judge_reason", "error",
    ]
    with (JOB / "per_question.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    body = ""
    for approach, item in totals.items():
        accuracy = item["accuracy_on_scored_percent"]
        body += (
            "<tr>"
            f"<td>{html.escape(approach)}</td>"
            f"<td>{item['scored']} / {item['eligible']}</td>"
            f"<td>{item['correct']}</td>"
            f"<td>{accuracy:.2f}%</td>" if accuracy is not None else
            "<tr>"
            f"<td>{html.escape(approach)}</td>"
            f"<td>{item['scored']} / {item['eligible']}</td>"
            f"<td>{item['correct']}</td><td>Unknown</td>"
        )
        body += (
            f"<td>{item['query_generation_tokens_reported']:,}</td>"
            f"<td>{item['query_embedding_tokens_reported']:,}</td>"
            f"<td>{item['judge_generation_tokens_reported']:,}</td></tr>"
        )

    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Full LoCoMo comparison</title>
<style>
body{{font:16px/1.6 system-ui;margin:40px;color:#18324b;background:#f5f7fb}}
table{{border-collapse:collapse;background:white;width:100%}}
td,th{{padding:14px;border-bottom:1px solid #dde5ee;text-align:left}}
th{{background:#18324b;color:white}}
pre{{white-space:pre-wrap;background:white;padding:20px}}
</style></head><body>
<h1>Full LoCoMo comparison</h1>
<p>Status: <strong>{report["status"]}</strong></p>
<p>10 conversations · Categories 1–4 · Custom judge rubric</p>
<p>Original development questions are included; this is not a pure held-out score.</p>
<table><tr><th>Approach</th><th>Scored / eligible</th><th>Correct</th>
<th>Accuracy on scored</th><th>Query generation tokens*</th>
<th>Query embedding tokens*</th><th>Judge tokens*</th></tr>{body}</table>
<p>*Reported usage. Check usage_complete in the JSON; unknown usage is not zero.</p>
<p>Build usage is separate. Retrieval budgets differ between implementations.</p>
<h2>Execution state</h2>
<pre>{html.escape(json.dumps(state, indent=2, ensure_ascii=False))}</pre>
</body></html>"""
    (JOB / "comparison.html").write_text(document, encoding="utf-8")
    return report


def main():
    load_dotenv(ROOT / ".env", override=True)

    raw = ROOT / "datasets/raw/locomo/locomo10.json"
    samples = read(raw)
    development = read(
        OUTPUTS / "benchmark-bc64ae07dd5a/config.json"
    )

    for variable, expected in (
        ("LLM_MODEL", development["answer_model"]),
        ("JUDGE_MODEL", development["judge_model"]),
        ("EMBEDDING_MODEL", development["embedding_model"]),
    ):
        if os.environ.get(variable) != expected:
            raise ValueError(f"{variable} differs from the existing experiment")

    if sha(raw) != development["raw_dataset_sha256"]:
        raise ValueError("Raw dataset differs from the existing experiment")

    protocol = {
        "dataset_sha256": sha(raw),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "categories": [1, 2, 3, 4],
        "answer_model": development["answer_model"],
        "judge_model": development["judge_model"],
        "embedding_model": development["embedding_model"],
        "judge_prompt": development["judge_prompt"],
        "development_question_ids": [
            question["question_id"]
            for question in development["questions"]
        ],
        "runner_hashes": {
            name: sha(ROOT / "scripts" / name)
            for name in (
                "run_memory_experiment.py",
                "run_agentic_rag.py",
                "build_offline_memory.py",
                "resume_offline_memory.py",
            )
        },
    }

    protocol_path = JOB / "protocol.json"
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError("Frozen full-benchmark protocol changed")
    save(protocol_path, protocol)

    runners = {
        "ensimem": create_cached_benchmark(
            "benchmark_memory.py", "benchmark_all_ensimem.py"
        ),
        "agentic_rag": create_cached_benchmark(
            "benchmark_agentic_rag.py", "benchmark_all_agentic.py"
        ),
    }

    state_path = JOB / "runs.json"
    state = read(state_path) if state_path.exists() else {}

    print("FULL DATASET:", len(samples), "conversations", flush=True)
    print("Output:", JOB, flush=True)

    for sample in samples:
        sample_id = sample["sample_id"]
        count = sum(
            q.get("category") in (1, 2, 3, 4) and "answer" in q
            for q in sample["qa"]
        )
        entry = state.setdefault(sample_id, {})

        if entry.get("status") == "completed":
            print("Already completed:", sample_id, flush=True)
            continue

        print(
            f"\n=== {sample_id}: {count} questions per approach ===",
            flush=True,
        )
        entry["status"] = "running"
        entry.pop("error", None)
        save(state_path, state)

        try:
            memory = build_memory(sample_id, protocol["dataset_sha256"])
            entry["memory"] = str(memory)
            save(state_path, state)

            for approach, script in runners.items():
                existing = entry.get(approach)
                if existing and (Path(existing) / "summary.json").exists():
                    old = read(Path(existing) / "summary.json")
                    if old["scored"] == count and old["failed"] == 0:
                        continue

                folder = benchmark(
                    script, memory, count,
                    JOB / f"{sample_id}-{approach}.log",
                )
                entry[approach] = str(folder)
                save(state_path, state)

            summaries = [
                read(Path(entry[approach]) / "summary.json")
                for approach in runners
            ]
            entry["status"] = (
                "completed"
                if all(
                    s["scored"] == count and s["failed"] == 0
                    for s in summaries
                )
                else "incomplete"
            )

        except Exception as error:
            entry["status"] = "incomplete"
            entry["error"] = str(error)
            print("INCOMPLETE:", sample_id, error, flush=True)

        finally:
            save(state_path, state)
            export(state, samples)

    report = export(state, samples)
    print("\nFULL BENCHMARK STATUS:", report["status"].upper())
    print(json.dumps(report["results"], indent=2))
    print("\nOpen report:")
    print(f'open "{JOB / "comparison.html"}"')


if __name__ == "__main__":
    main()
