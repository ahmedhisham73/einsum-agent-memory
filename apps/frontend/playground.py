"""Local interactive explorer for the existing experiment artifacts."""

import json
import subprocess
import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MEMORY = ROOT / "outputs/offline-memory-a5e48166f56e"

st.set_page_config(
    page_title="Agent Memory Lab",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.stApp {background:#f5f7fb}
h1,h2,h3 {color:#15314c}
[data-testid="stMetric"] {
    background:white;border:1px solid #dce5ee;
    padding:18px;border-radius:12px;
}
div.stButton > button {
    background:#087f79;color:white;border-radius:8px;
}
</style>
""", unsafe_allow_html=True)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def show_episode(episode, highlighted=()):
    st.caption(
        f'{episode["episode_id"]} · '
        f'{episode.get("source_timestamp", episode["observed_at"])}'
    )
    for turn in episode["turns"]:
        label = f'[{turn["turn_id"]}] {turn["speaker"]}'
        if turn["turn_id"] in highlighted:
            st.info(label + "\n\n" + turn["text"])
        else:
            st.markdown("**" + label + "**")
            st.write(turn["text"])
        if turn.get("metadata"):
            with st.expander("Source media metadata · no image fetch"):
                st.json(turn["metadata"])


def show_run(folder):
    result = read(folder / "result.json")
    evidence = read(folder / "evidence.json")
    usage = read(folder / "usage.json")

    st.subheader(result["question"])
    st.caption(f"Saved run: {folder.name}")

    generation = [u for u in usage if u["usage_kind"] == "generation"]
    embeddings = [u for u in usage if u["usage_kind"] == "embedding"]

    def total(items):
        if any(
            u.get("input_tokens") is None or u.get("output_tokens") is None
            for u in items
        ):
            return "Unknown"
        return f'{sum(u["input_tokens"] + u["output_tokens"] for u in items):,}'

    columns = st.columns(4)
    columns[0].metric("Source episodes supplied", len(evidence))
    columns[1].metric("Generation tokens", total(generation))
    columns[2].metric("Embedding tokens", total(embeddings))
    columns[3].metric("Run time", f'{result["elapsed_seconds"]:.2f} s')

    st.success(result["answer"])
    st.caption(
        "Evidence sufficient: "
        + str(result["sufficient"])
        + " — model assessment, not an independent accuracy score."
    )

    plan_tab, evidence_tab, usage_tab, trace_tab = st.tabs([
        "1 · Retrieval plan",
        "2 · Original evidence",
        "3 · Token accounting",
        "4 · Execution trace",
    ])

    with plan_tab:
        plan_path = folder / "plan.json"
        if plan_path.exists():
            st.json(read(plan_path))
        retrieval_path = folder / "retrieval.json"
        if retrieval_path.exists():
            st.write("Retrieval rounds")
            for index, round_data in enumerate(read(retrieval_path), 1):
                st.write(
                    f'Round {index}: '
                    f'{len(round_data["selected_episode_ids"])} episodes'
                )

    with evidence_tab:
        st.write("Answer citations")
        st.json(result["citations"])
        for episode in evidence:
            highlighted = [
                citation["turn_id"]
                for citation in result["citations"]
                if citation["episode_id"] == episode["episode_id"]
            ]
            with st.expander(
                episode.get("theme", episode["episode_id"]),
                expanded=bool(highlighted),
            ):
                show_episode(episode, highlighted)

    with usage_tab:
        st.dataframe([
            {
                "stage": u["stage"],
                "kind": u["usage_kind"],
                "model": u["model"],
                "input": u["input_tokens"],
                "output": u["output_tokens"],
                "cached_input": u["cached_input_tokens"],
            }
            for u in usage
        ], use_container_width=True)
        st.caption("Cached input is already included in input tokens.")

    with trace_tab:
        trace_path = folder / "traces.json"
        if trace_path.exists():
            for trace in read(trace_path):
                with st.expander(trace["stage"]):
                    st.json(trace)
        st.download_button(
            "Download result JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name=folder.name + ".json",
            mime="application/json",
        )


st.title("🧠 Agent Memory Lab")
st.caption("Explore the memory. Follow the retrieval. Inspect the evidence.")

if not (MEMORY / "records.json").exists():
    st.error("Memory folder not found: " + str(MEMORY))
    st.stop()

records = read(MEMORY / "records.json")
episodes = read(MEMORY / "episodes.json")
source = read(MEMORY / "source_conversation.json")
episode_map = {e["episode_id"]: e for e in episodes}

st.sidebar.header("Current memory")
st.sidebar.write("LoCoMo · conv-26")
st.sidebar.metric("Sessions", len(source["sessions"]))
st.sidebar.metric("Episodes", len(episodes))
st.sidebar.metric("Index records", len(records))
st.sidebar.caption(
    "Existing memory snapshot. New questions do not update stored memory."
)

offline_tab, online_tab, results_tab = st.tabs([
    "📚 Offline Memory", "🔎 Online Query", "📊 Results"
])

with offline_tab:
    st.subheader("An index record is an address to source evidence")
    entity = st.selectbox(
        "Choose an entity",
        ["All entities"] + sorted({r["entity"] for r in records}),
    )
    search = st.text_input("Filter by property or value")
    filtered = [
        r for r in records
        if (entity == "All entities" or r["entity"] == entity)
        and search.casefold() in (
            r["property"] + " " + r["value"]
        ).casefold()
    ]

    if filtered:
        selected = st.selectbox(
            "Choose a record",
            range(len(filtered)),
            format_func=lambda i: (
                f'{filtered[i]["entity"]} · '
                f'{filtered[i]["property"]}: {filtered[i]["value"]}'
            ),
        )
        record = filtered[selected]
        left, right = st.columns([1, 2])
        with left:
            st.markdown("### Structured index record")
            st.json(record)
        with right:
            st.markdown("### Original supporting episode")
            show_episode(
                episode_map[record["episode_id"]],
                record["evidence_turn_ids"],
            )
    else:
        st.info("No records match this filter.")

with online_tab:
    mode = st.radio(
        "Explore online memory access",
        ["Saved experiment · no API calls", "Run a new question"],
        horizontal=True,
    )

    if mode.startswith("Saved"):
        candidates = list((ROOT / "outputs").glob("query-*"))
        candidates += list(
            (ROOT / "outputs").glob("benchmark-*/q*-query")
        )
        candidates += list((ROOT / "outputs").glob("ui-query-*"))

        saved = sorted({
            path for path in candidates
            if (path / "result.json").exists()
            and (path / "plan.json").exists()
            and (path / "evidence.json").exists()
        })

        if saved:
            chosen = st.selectbox(
                "Select a saved question",
                saved,
                format_func=lambda p: (
                    read(p / "result.json")["question"] + " · " + p.name
                ),
            )
            show_run(chosen)
        else:
            st.info("No saved EnSIMem query runs found.")

    else:
        st.caption(
            "Run uses the real API. Existing embeddings are reused. "
            "Each submission creates a separate saved run."
        )
        with st.form("new_question"):
            question = st.text_input(
                "Question about this conversation",
                value="What did Caroline research?",
            )
            submitted = st.form_submit_button("Run memory query")

        if submitted:
            if not question.strip():
                st.error("Enter a question.")
            else:
                folder = ROOT / "outputs" / (
                    "ui-query-" + uuid.uuid4().hex[:12]
                )
                st.session_state["last_live_run"] = None

                with st.status(
                    "Running online memory access…", expanded=True
                ) as status:
                    log_box = st.empty()
                    lines = []
                    process = subprocess.Popen(
                        [
                            sys.executable, "-u",
                            str(ROOT / "scripts/run_memory_experiment.py"),
                            "--memory", str(MEMORY),
                            "--question", question.strip(),
                            "--output-dir", str(folder),
                        ],
                        cwd=ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    for line in process.stdout:
                        lines.append(line.rstrip())
                        log_box.code("\n".join(lines[-12:]))
                    process.wait()

                    folder.mkdir(exist_ok=True)
                    (folder / "ui_run.log").write_text(
                        "\n".join(lines), encoding="utf-8"
                    )

                    if process.returncode == 0:
                        st.session_state["last_live_run"] = str(folder)
                        status.update(
                            label="Query completed",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        status.update(
                            label="Run failed · log preserved",
                            state="error",
                        )

        if st.session_state.get("last_live_run"):
            show_run(Path(st.session_state["last_live_run"]))

with results_tab:
    folders = [
        ("EnSIMem prototype", ROOT / "outputs/benchmark-bc64ae07dd5a"),
        ("Agentic RAG", ROOT / "outputs/benchmark-agentic-81d4d0e869b2"),
    ]
    rows = []
    for label, folder in folders:
        if not (folder / "summary.json").exists():
            continue
        summary = read(folder / "summary.json")
        generation = [
            row for row in summary["query_usage"]
            if row["usage_kind"] == "generation"
        ]
        values = [row["total_tokens"] for row in generation]
        rows.append({
            "Approach": label,
            "Correct": summary["correct"],
            "Scored": summary["scored"],
            "Accuracy %": summary["accuracy_on_scored_percent"],
            "Query generation tokens": (
                sum(values) if all(v is not None for v in values) else None
            ),
        })

    st.subheader("Saved 10-question development experiment")
    st.dataframe(rows, use_container_width=True)
    st.caption(
        "Custom judge rubric, one conversation, different retrieval budgets. "
        "Build and judge usage are excluded from the token column."
    )
