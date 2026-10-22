"""Benchmark the largest completed constructed memory."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

st.set_page_config(
    page_title="Agent Memory Benchmark Lab",
    page_icon="🧠",
    layout="wide",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def largest_completed_memory():
    candidates = []

    for folder in OUTPUTS.glob("offline-memory-*"):
        if "-backup-" in folder.name:
            continue

        try:
            manifest = read(folder / "manifest.json")
            if manifest.get("status") != "completed":
                continue

            records = read(folder / "records.json")
            episodes = read(folder / "episodes.json")

            if not isinstance(records, list) or not isinstance(episodes, list):
                continue

            candidates.append({
                "folder": folder,
                "manifest": manifest,
                "records": records,
                "episodes": episodes,
            })
        except (OSError, ValueError):
            continue

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            len(item["records"]),
            len(item["episodes"]),
            item["folder"].name,
        ),
    )


def discover_runs(manifest, agentic):
    runs = []

    for folder in OUTPUTS.glob("benchmark-*"):
        if folder.name.startswith("benchmark-agentic-") != agentic:
            continue

        try:
            config = read(folder / "config.json")
        except (OSError, ValueError):
            continue

        if (
            config.get("memory_build") == manifest.get("experiment_id")
            and config.get("sample_id") == manifest.get("conversation_id")
        ):
            runs.append((folder, config))

    return sorted(
        runs,
        key=lambda item: (item[0] / "config.json").stat().st_mtime,
        reverse=True,
    )


def load_reports(folder, config):
    reports = {}

    for question in config.get("questions", []):
        qid = question["question_id"]
        path = folder / (qid.split("::")[-1] + ".json")

        if path.exists():
            reports[qid] = read(path)

    return reports


def query_folder(folder, qid):
    return folder / (qid.split("::")[-1] + "-query")


def token_usage(folder, qid):
    """Calculate real reported query usage; missing values stay unknown."""
    path = query_folder(folder, qid) / "usage.json"

    if not path.exists():
        return None

    entries = read(path)
    if not isinstance(entries, list) or not entries:
        return None

    unique = {}

    for entry in entries:
        call_id = entry.get("call_id")

        if not call_id:
            return None

        if call_id in unique and unique[call_id] != entry:
            return None

        unique[call_id] = entry

    incoming = 0
    outgoing = 0

    for entry in unique.values():
        input_tokens = entry.get("input_tokens")
        output_tokens = entry.get("output_tokens")

        if (
            not isinstance(input_tokens, int)
            or not isinstance(output_tokens, int)
            or input_tokens < 0
            or output_tokens < 0
        ):
            return None

        incoming += input_tokens
        outgoing += output_tokens

    return {
        "Input": incoming,
        "Output": outgoing,
        "Total": incoming + outgoing,
        "Calls": len(unique),
    }


def run_label(item):
    folder, config = item

    return (
        f'{folder.name} · {len(config.get("questions", []))} questions'
        f' · answer={config.get("answer_model")}'
        f' · judge={config.get("judge_model")}'
    )


def comparison_row(label, folder, reports, ids):
    usages = [token_usage(folder, qid) for qid in ids]
    complete_usage = all(item is not None for item in usages)

    correct = sum(
        reports[qid].get("label") == "CORRECT"
        for qid in ids
    )

    return {
        "Approach": label,
        "Paired questions": len(ids),
        "Accuracy %": 100 * correct / len(ids),
        "Query input tokens": (
            sum(item["Input"] for item in usages)
            if complete_usage else None
        ),
        "Query output tokens": (
            sum(item["Output"] for item in usages)
            if complete_usage else None
        ),
        "Query total tokens": (
            sum(item["Total"] for item in usages)
            if complete_usage else None
        ),
        "Mean tokens / question": (
            sum(item["Total"] for item in usages) / len(ids)
            if complete_usage else None
        ),
        "Questions with unknown usage": sum(
            item is None for item in usages
        ),
    }


st.title("🧠 Agent Memory Benchmark Lab")

selected = largest_completed_memory()

if selected is None:
    st.error("No valid completed offline memory found.")
    st.stop()

memory = selected["folder"]
manifest = selected["manifest"]
records = selected["records"]
episodes = selected["episodes"]

st.caption(
    "Automatically uses the largest completed memory by record count."
)

st.sidebar.header("Active constructed memory")
st.sidebar.write(manifest["conversation_id"])
st.sidebar.caption(memory.name)
st.sidebar.metric("Index records", len(records))
st.sidebar.metric("Episodes", len(episodes))

if st.sidebar.button("Refresh results"):
    st.rerun()

compare_tab, memory_tab, paper_tab, run_tab = st.tabs([
    "📊 Compare results",
    "📚 Constructed memory",
    "📄 Paper reference",
    "▶ Run benchmarks",
])

with compare_tab:
    ensimem_runs = discover_runs(manifest, agentic=False)
    agentic_runs = discover_runs(manifest, agentic=True)

    if not ensimem_runs or not agentic_runs:
        st.info(
            "Saved results for both approaches on this memory are required. "
            "Use the Run benchmarks tab."
        )
    else:
        left, right = st.columns(2)

        with left:
            ei = st.selectbox(
                "EnSIMem result run",
                range(len(ensimem_runs)),
                format_func=lambda i: run_label(ensimem_runs[i]),
            )

        with right:
            ai = st.selectbox(
                "Agentic RAG result run",
                range(len(agentic_runs)),
                format_func=lambda i: run_label(agentic_runs[i]),
            )

        ef, ec = ensimem_runs[ei]
        af, ac = agentic_runs[ai]

        er = load_reports(ef, ec)
        ar = load_reports(af, ac)

        compatibility_fields = (
            "memory_build",
            "raw_dataset_sha256",
            "answer_model",
            "judge_model",
            "embedding_model",
            "judge_prompt",
            "base_url",
            "timeout_seconds",
            "gateway_sha256",
        )

        differences = [
            key for key in compatibility_fields
            if ec.get(key) is None or ec.get(key) != ac.get(key)
        ]

        if differences:
            st.warning(
                "Settings differ or are missing: "
                + ", ".join(differences)
                + ". Treat this comparison as exploratory."
            )

        coverage = []

        for label, config, saved in (
            ("EnSIMem", ec, er),
            ("Agentic RAG", ac, ar),
        ):
            total = len(config.get("questions", []))
            scored = sum(
                report.get("status") == "scored"
                for report in saved.values()
            )
            failed = sum(
                report.get("status") == "failed"
                for report in saved.values()
            )

            coverage.append({
                "Approach": label,
                "Selected": total,
                "Scored": scored,
                "Failed": failed,
                "Pending": total - len(saved),
                "Completion %": (
                    100 * scored / total if total else None
                ),
            })

        st.subheader("Evaluation coverage")
        st.dataframe(
            coverage,
            hide_index=True,
            use_container_width=True,
        )

        paired = sorted(
            qid for qid in er.keys() & ar.keys()
            if er[qid].get("status") == ar[qid].get("status") == "scored"
            and er[qid].get("label") in ("CORRECT", "WRONG")
            and ar[qid].get("label") in ("CORRECT", "WRONG")
            and all(
                er[qid].get(key) == ar[qid].get(key)
                for key in ("question", "gold_answer", "category")
            )
        )

        st.caption(
            f"Common matching questions scored by both: {len(paired)}"
        )

        if paired:
            rows = [
                comparison_row("EnSIMem", ef, er, paired),
                comparison_row("Agentic RAG", af, ar, paired),
            ]
            frame = pd.DataFrame(rows).set_index("Approach")

            accuracy_column, tokens_column = st.columns(2)

            with accuracy_column:
                st.subheader("Paired accuracy")
                st.bar_chart(frame[["Accuracy %"]])

            with tokens_column:
                st.subheader("Actual query API tokens")
                known = frame[["Query total tokens"]].dropna()

                if not known.empty:
                    st.bar_chart(known)

                if len(known) != len(frame):
                    st.info(
                        "Unknown token totals are omitted from the chart."
                    )

            st.dataframe(frame, use_container_width=True)
            st.caption(
                "Token totals are calculated from saved usage.json files "
                "for the paired questions. Generation and embedding calls "
                "are included. Build and judge usage are excluded. "
                "Cached input is already included in input tokens. "
                "Missing usage remains unknown."
            )

            category_rows = []

            for category in (1, 2, 3, 4):
                ids = [
                    qid for qid in paired
                    if er[qid].get("category") == category
                ]

                if not ids:
                    continue

                row = {
                    "Category": category,
                    "Paired questions": len(ids),
                }

                for label, saved in (
                    ("EnSIMem", er),
                    ("Agentic RAG", ar),
                ):
                    row[label + " accuracy %"] = 100 * sum(
                        saved[qid]["label"] == "CORRECT"
                        for qid in ids
                    ) / len(ids)

                category_rows.append(row)

            st.subheader("Accuracy by category")
            st.dataframe(category_rows, hide_index=True)

            qid = st.selectbox(
                "Inspect a question",
                paired,
                format_func=lambda q: q + " · " + er[q]["question"],
            )

            st.write("Reference answer")
            st.write(er[qid].get("gold_answer"))

            for column, label, folder, saved in zip(
                st.columns(2),
                ("EnSIMem", "Agentic RAG"),
                (ef, af),
                (er, ar),
            ):
                with column:
                    report = saved[qid]

                    st.subheader(label)
                    st.write(report.get("generated_answer", ""))
                    st.write("Verdict:", report.get("label"))
                    st.caption(report.get("judge_reason", ""))

                    usage = token_usage(folder, qid)

                    if usage is None:
                        st.write("Actual API tokens: Unknown")
                    else:
                        st.metric(
                            "Actual API tokens",
                            f'{usage["Total"]:,}',
                        )
                        st.write(usage)

                    with st.expander("Citations and saved artifacts"):
                        st.json(report.get("citations", []))

                        for name in (
                            "usage.json",
                            "evidence.json",
                            "plan.json",
                            "retrieval.json", "coverage.json", "candidate_matches.json",
                            "result.json",
                        ):
                            path = query_folder(folder, qid) / name

                            if path.exists():
                                st.write(name)
                                st.json(read(path))

            st.download_button(
                "Download paired comparison CSV",
                pd.DataFrame(rows).to_csv(index=False),
                file_name="paired_comparison.csv",
                mime="text/csv",
            )
        else:
            st.info("No common scored questions yet.")

        with st.expander("Failed questions"):
            failures = []

            for label, saved in (
                ("EnSIMem", er),
                ("Agentic RAG", ar),
            ):
                for qid, report in saved.items():
                    if report.get("status") == "failed":
                        failures.append({
                            "Approach": label,
                            "Question ID": qid,
                            "Error": report.get("error"),
                        })

            if failures:
                st.dataframe(failures, hide_index=True)
            else:
                st.write("No saved failures in the selected runs.")

with memory_tab:
    st.subheader("Largest completed constructed memory")
    st.write("Conversation:", manifest["conversation_id"])
    st.write("Folder:", memory.name)

    search = st.text_input("Search entity, property or value")

    filtered = [
        record for record in records
        if search.casefold() in " ".join(
            str(record.get(key, ""))
            for key in ("entity", "property", "value")
        ).casefold()
    ]

    st.caption(f"{len(filtered)} matching records")

    if filtered:
        index = st.selectbox(
            "Record",
            range(len(filtered)),
            format_func=lambda i: (
                f'{filtered[i].get("entity")} · '
                f'{filtered[i].get("property")}: '
                f'{filtered[i].get("value")}'
            ),
        )

        record = filtered[index]
        st.json(record)

        episode = next(
            (
                item for item in episodes
                if item["episode_id"] == record.get("episode_id")
            ),
            None,
        )

        if episode:
            st.subheader("Original supporting episode")
            st.json(episode)

with paper_tab:
    st.subheader("Published LoCoMo reference results")

    st.link_button(
        "Open EnSIMem paper",
        "https://arxiv.org/html/2609.27279v1",
    )

    st.info(
        "These are published reference scores from Table 1. "
        "Local results use the repository's custom judge rubric "
        "and the selected conversation."
    )

    paper = pd.DataFrame([
        ["Full Context", 82.5],
        ["RAG", 63.3],
        ["HippoRAG", 47.1],
        ["Zep*", 61.6],
        ["Mem0", 65.3],
        ["LangMem*", 73.4],
        ["Nemori*", 79.4],
        ["MEMORA (S)", 84.9],
        ["MEMORA (P)", 86.3],
        ["EnSIMem · GPT-4o-mini judge", 90.6],
        ["EnSIMem · Qwen3-32B judge", 90.0],
    ], columns=["Method", "Overall accuracy %"])

    st.bar_chart(paper.set_index("Method"))
    st.dataframe(
        paper,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Baseline scores were transcribed by the paper's authors "
        "from Memora; * marks prior-work baselines. "
        "The paper's RAG baseline differs from our Agentic RAG."
    )

    st.write(
        "Paper Table 3 reports a mean context size of 7,470 tokens. "
        "This published context metric is separate from actual API "
        "consumption calculated in the Compare results tab."
    )

with run_tab:
    st.subheader("Run on the largest completed memory")
    st.write("Conversation:", manifest["conversation_id"])
    st.write("Memory folder:", memory.name)

    st.caption(
        "Reuses the constructed offline memory. "
        "Clicking Run makes real API calls for benchmarking. "
        "Compatible saved results may be reused by the existing runners."
    )

    st.info(
        "The existing runners may skip saved failed reports and stop "
        "individual questions with partial query artifacts. "
        "Inspect Scored, Failed and Pending after execution."
    )

    dataset_path = ROOT / "datasets/raw/locomo/locomo10.json"

    if not dataset_path.exists():
        st.error("Local LoCoMo dataset not found.")
    else:
        sample = next(
            (
                item for item in read(dataset_path)
                if item["sample_id"] == manifest["conversation_id"]
            ),
            None,
        )

        if sample is None:
            st.error("Conversation not found in the local dataset.")
        else:
            count = sum(
                question.get("category") in (1, 2, 3, 4)
                and "answer" in question
                for question in sample["qa"]
            )

            if not count:
                st.info("No eligible questions found.")
            else:
                limit = st.number_input(
                    "Questions per approach",
                    min_value=1,
                    max_value=count,
                    value=count,
                    step=1,
                )

                if st.button("Run EnSIMem + Agentic RAG"):
                    for script in (
                        "benchmark_all_ensimem.py",
                        "benchmark_all_agentic.py",
                    ):
                        st.write("Running:", script)
                        log_box = st.empty()
                        tail = []

                        with subprocess.Popen(
                            [
                                sys.executable,
                                "-u",
                                str(ROOT / "scripts" / script),
                                "--memory",
                                str(memory),
                                "--limit",
                                str(int(limit)),
                            ],
                            cwd=ROOT,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        ) as process:
                            for line in process.stdout:
                                tail = (tail + [line.rstrip()])[-20:]
                                log_box.code("\n".join(tail))

                            exit_code = process.wait()

                        if exit_code:
                            st.error(
                                f"{script} exited with code {exit_code}."
                            )
                            break
                    else:
                        st.success(
                            "Both runners finished. Click Refresh results "
                            "to load the saved comparison."
                        )
