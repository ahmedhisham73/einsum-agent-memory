"""Export saved experiment results. No API calls or extra dependencies."""

import argparse
import csv
import html
import json
import math
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def text(value):
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False
    )


def known_sum(records, field):
    if not records:
        return 0
    values = [record.get(field) for record in records]
    return sum(values) if all(v is not None for v in values) else None


def token_total(records):
    input_tokens = known_sum(records, "input_tokens")
    output_tokens = known_sum(records, "output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    return input_tokens + output_tokens


def kind_total(records, kind):
    return token_total([
        record for record in records
        if record["usage_kind"] == kind
    ])


def rollup(records):
    groups = {}
    for record in records:
        key = (
            record["phase"],
            record["stage"],
            record["usage_kind"],
            record["model"],
        )
        groups.setdefault(key, []).append(record)

    return [
        {
            "phase": key[0],
            "stage": key[1],
            "usage_kind": key[2],
            "model": key[3],
            "calls": len(items),
            "input_tokens": known_sum(items, "input_tokens"),
            "output_tokens": known_sum(items, "output_tokens"),
            "cached_input_tokens": known_sum(items, "cached_input_tokens"),
            "total_tokens": token_total(items),
            "missing_usage_calls": sum(
                item.get("input_tokens") is None
                or item.get("output_tokens") is None
                for item in items
            ),
        }
        for key, items in sorted(groups.items())
    ]


def latency_stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def load_benchmark(folder, approach):
    config = read(folder / "config.json")
    questions = []
    usage = []
    results = []

    for item in config["questions"]:
        short_id = item["question_id"].split("::")[-1]
        report = read(folder / f"{short_id}.json")
        query_dir = folder / f"{short_id}-query"
        result_path = query_dir / "result.json"
        result = read(result_path) if result_path.exists() else {}

        if report["question_id"] != item["question_id"]:
            raise ValueError("Question identity mismatch")
        if report["question"] != item["question"]:
            raise ValueError("Question text mismatch")

        usage_path = query_dir / "usage.json"
        query_usage = read(usage_path) if usage_path.exists() else []
        usage.extend(query_usage)
        results.append(result)

        questions.append({
            "question_id": item["question_id"],
            "category": item["category"],
            "question": item["question"],
            "gold_answer": text(item["gold_answer"]),
            "status": report["status"],
            "label": report.get("label"),
            "answer": report.get("generated_answer", ""),
            "judge_reason": report.get("judge_reason", ""),
            "error": report.get("error", ""),
            "generation_tokens": kind_total(query_usage, "generation"),
            "embedding_tokens": kind_total(query_usage, "embedding"),
            "generation_calls": sum(
                record["usage_kind"] == "generation"
                for record in query_usage
            ),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "citations": report.get("citations", []),
        })

    judge_path = folder / "judge_usage.json"
    judge_usage = read(judge_path) if judge_path.exists() else []

    scored = [q for q in questions if q["status"] == "scored"]
    correct = sum(q["label"] == "CORRECT" for q in scored)

    return {
        "approach": approach,
        "folder": str(folder),
        "config": config,
        "questions": questions,
        "query_usage": usage,
        "judge_usage": judge_usage,
        "query_results": results,
        "selected": len(questions),
        "scored": len(scored),
        "failed": len(questions) - len(scored),
        "correct": correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ensimem", default="outputs/benchmark-bc64ae07dd5a"
    )
    parser.add_argument(
        "--agentic", default="outputs/benchmark-agentic-81d4d0e869b2"
    )
    parser.add_argument(
        "--memory", default="outputs/offline-memory-a5e48166f56e"
    )
    args = parser.parse_args()

    ensimem = load_benchmark(Path(args.ensimem).resolve(), "ensimem")
    agentic = load_benchmark(Path(args.agentic).resolve(), "agentic_rag")
    memory = Path(args.memory).resolve()
    memory_manifest = read(memory / "manifest.json")

    for field in (
        "questions", "raw_dataset_sha256",
        "answer_model", "judge_model", "judge_prompt",
    ):
        if ensimem["config"][field] != agentic["config"][field]:
            raise ValueError(f"Comparison settings differ: {field}")

    for experiment in (ensimem, agentic):
        if (
            experiment["config"]["memory_build"]
            != memory_manifest["experiment_id"]
        ):
            raise ValueError("Memory build does not match benchmark")

    # Resolve exact build artifacts from the saved query results.
    dense_ids = {
        result["embedding_fingerprint"][:12]
        for result in ensimem["query_results"]
        if "embedding_fingerprint" in result
    }
    baseline_ids = {
        result["baseline_build"]
        for result in agentic["query_results"]
        if "baseline_build" in result
    }

    if len(dense_ids) != 1 or len(baseline_ids) != 1:
        raise ValueError("Expected one identifiable build per approach")

    dense_dir = memory / ("dense-" + next(iter(dense_ids)))
    baseline_dir = ROOT / "outputs" / next(iter(baseline_ids))

    ensimem["build_usage"] = (
        read(memory / "usage.json")
        + read(dense_dir / "cache.json")["usage"]
    )
    agentic["build_usage"] = read(
        baseline_dir / "cache.json"
    )["usage"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "outputs/exports" / (
        f"comparison-{stamp}-{uuid.uuid4().hex[:6]}"
    )
    output.mkdir(parents=True)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": ensimem["config"]["sample_id"],
        "scope": "10-question development slice; custom LLM judge rubric",
        "notes": [
            "Accuracy is calculated on scored questions; failures shown separately.",
            "Build, query and judge usage are reported separately.",
            "Generation and embedding tokens use different models and prices.",
            "Cached input tokens are a subset of input, not extra tokens.",
            "Latency uses saved runner elapsed time, including local preparation.",
            "Single observations per question; not a latency confidence estimate.",
            "Implementations have different retrieval budgets and policies.",
            "No dollar costs are inferred from token totals.",
        ],
        "approaches": {},
    }

    all_usage = []
    call_keys = {}

    for experiment in (ensimem, agentic):
        approach = experiment["approach"]

        for scope, key in (
            ("build", "build_usage"),
            ("query", "query_usage"),
            ("evaluation", "judge_usage"),
        ):
            unique = []
            for record in experiment[key]:
                call_key = (approach, record["call_id"])
                previous = call_keys.get(call_key)
                if previous is not None:
                    if previous != (scope, record):
                        raise ValueError("Conflicting or misattributed call ID")
                    continue
                call_keys[call_key] = (scope, record)
                unique.append(record)
                all_usage.append({
                    **record,
                    "scope": scope,
                    "approach": approach,
                })
            experiment[key] = unique

        scored_count = experiment["scored"]
        latencies = [
            q["elapsed_seconds"] for q in experiment["questions"]
            if q["elapsed_seconds"] is not None
        ]

        summary["approaches"][approach] = {
            "selected": experiment["selected"],
            "scored": scored_count,
            "failed": experiment["failed"],
            "correct": experiment["correct"],
            "accuracy_on_scored_percent": (
                100 * experiment["correct"] / scored_count
                if scored_count else None
            ),
            "build_generation_tokens": kind_total(
                experiment["build_usage"], "generation"
            ),
            "build_embedding_tokens": kind_total(
                experiment["build_usage"], "embedding"
            ),
            "query_generation_tokens": kind_total(
                experiment["query_usage"], "generation"
            ),
            "query_embedding_tokens": kind_total(
                experiment["query_usage"], "embedding"
            ),
            "judge_generation_tokens": kind_total(
                experiment["judge_usage"], "generation"
            ),
            "latency": latency_stats(latencies),
            "build_breakdown": rollup(experiment["build_usage"]),
            "query_breakdown": rollup(experiment["query_usage"]),
            "judge_breakdown": rollup(experiment["judge_usage"]),
        }

    agentic_questions = {
        q["question_id"]: q for q in agentic["questions"]
    }
    paired = []

    for left in ensimem["questions"]:
        right = agentic_questions[left["question_id"]]
        row = {
            key: left[key]
            for key in ("question_id", "category", "question", "gold_answer")
        }
        for name, item in (("ensimem", left), ("agentic", right)):
            for field in (
                "status", "label", "answer", "judge_reason", "error",
                "generation_tokens", "embedding_tokens",
                "generation_calls", "elapsed_seconds",
            ):
                row[f"{name}_{field}"] = item[field]
            row[f"{name}_citations"] = json.dumps(
                item["citations"], ensure_ascii=False
            )
        paired.append(row)

    write_json(output / "summary.json", summary)
    write_json(output / "per_question.json", paired)
    write_json(output / "provenance.json", {
        "ensimem_benchmark": ensimem["folder"],
        "agentic_benchmark": agentic["folder"],
        "memory_build": str(memory),
        "ensimem_dense_build": str(dense_dir),
        "agentic_chunk_build": str(baseline_dir),
        "ensimem_config": ensimem["config"],
        "agentic_config": agentic["config"],
    })

    write_csv(
        output / "per_question.csv",
        paired,
        list(paired[0]),
    )
    usage_fields = [
        "approach", "scope", "experiment_id", "query_id", "call_id",
        "phase", "stage", "usage_kind", "provider", "model",
        "input_tokens", "output_tokens", "cached_input_tokens", "attempt",
    ]
    write_csv(output / "api_usage.csv", all_usage, usage_fields)

    def esc(value):
        return html.escape(str(value))

    def format_value(value):
        if value is None:
            return "Unknown"
        if isinstance(value, float):
            return f"{value:,.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    metrics = [
        ("Scored / selected", "scored"),
        ("Correct answers", "correct"),
        ("Failed", "failed"),
        ("Accuracy on scored (%)", "accuracy_on_scored_percent"),
        ("Build: generation tokens", "build_generation_tokens"),
        ("Build: embedding tokens", "build_embedding_tokens"),
        ("Query: generation tokens", "query_generation_tokens"),
        ("Query: embedding tokens", "query_embedding_tokens"),
        ("Judge: generation tokens", "judge_generation_tokens"),
    ]

    metric_rows = ""
    for label, key in metrics:
        values = []
        for approach in ("ensimem", "agentic_rag"):
            item = summary["approaches"][approach]
            value = (
                f'{item["scored"]} / {item["selected"]}'
                if key == "scored"
                else format_value(item[key])
            )
            values.append(f"<td>{esc(value)}</td>")
        metric_rows += f"<tr><th>{esc(label)}</th>{''.join(values)}</tr>"

    latency_rows = ""
    for label, key in (
        ("Mean runtime (s)", "mean_seconds"),
        ("Median runtime (s)", "median_seconds"),
        ("P95 runtime (s)", "p95_seconds"),
    ):
        cells = []
        for approach in ("ensimem", "agentic_rag"):
            stats = summary["approaches"][approach]["latency"] or {}
            cells.append(f"<td>{esc(format_value(stats.get(key)))}</td>")
        latency_rows += f"<tr><th>{label}</th>{''.join(cells)}</tr>"

    question_cards = ""
    for row in paired:
        columns = ""
        for prefix, title in (
            ("ensimem", "EnSIMem prototype"),
            ("agentic", "Agentic RAG"),
        ):
            label = row[f"{prefix}_label"] or row[f"{prefix}_status"]
            badge = "correct" if label == "CORRECT" else "other"
            columns += f"""
            <article>
              <h3>{title} <span class="{badge}">{esc(label)}</span></h3>
              <p>{esc(row[f"{prefix}_answer"])}</p>
              <p class="small">Generation tokens:
                {esc(format_value(row[f"{prefix}_generation_tokens"]))}
                · Runtime:
                {esc(format_value(row[f"{prefix}_elapsed_seconds"]))} s
              </p>
              <details><summary>Judge reasoning and citations</summary>
                <p>{esc(row[f"{prefix}_judge_reason"])}</p>
                <pre>{esc(row[f"{prefix}_citations"])}</pre>
                <p>{esc(row[f"{prefix}_error"])}</p>
              </details>
            </article>"""

        question_cards += f"""
        <section class="question">
          <p class="small">{esc(row["question_id"])}
             · Category {esc(row["category"])}</p>
          <h2>{esc(row["question"])}</h2>
          <p><strong>Reference:</strong> {esc(row["gold_answer"])}</p>
          <div class="pair">{columns}</div>
        </section>"""

    notes = "".join(f"<li>{esc(note)}</li>" for note in summary["notes"])

    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Memory Experiment Report</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;background:#f4f7fb;
color:#18324b;margin:0}
main{max-width:1180px;margin:40px auto;padding:0 24px}
h1{font-size:36px;line-height:1.2}h2{font-size:21px}h3{font-size:17px}
.tag{color:#087f79;font-weight:700;letter-spacing:1px}
table{width:100%;border-collapse:collapse;background:white;margin:24px 0}
td,th{padding:13px;text-align:left;border-bottom:1px solid #dde5ed}
thead{background:#15314c;color:white}
.question{background:white;border:1px solid #dde5ed;border-radius:12px;
padding:24px;margin:24px 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}
article{background:#f5f8fc;padding:18px;border-radius:8px;min-width:0}
.small{font-size:13px;color:#53677c}
.correct{color:#087f79}.other{color:#ab4b20}
span{font-size:12px;margin-left:10px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
details{margin-top:16px}
@media(max-width:750px){.pair{grid-template-columns:1fr}main{padding:0 12px}}
@media print{body{background:white}details{display:block}.question{break-inside:avoid}}
</style></head><body><main>
<p class="tag">AGENT MEMORY EXPERIMENT</p>
<h1>EnSIMem prototype vs Agentic RAG</h1>
<p>Same 10 questions · LoCoMo conv-26 · Custom LLM judge rubric</p>
<table><thead><tr><th>Metric</th><th>EnSIMem prototype</th>
<th>Agentic RAG</th></tr></thead><tbody>
""" + metric_rows + latency_rows + """
</tbody></table>
<details><summary>Scope and accounting notes</summary><ul>
""" + notes + """
</ul></details>
<h2>Question-by-question results</h2>
""" + question_cards + """
</main></body></html>
"""
    (output / "comparison.html").write_text(document, encoding="utf-8")

    print("\nExport completed:")
    for path in sorted(output.iterdir()):
        print(path)
    print("\nOpen the report:")
    print(f'open "{output / "comparison.html"}"')


if __name__ == "__main__":
    main()
