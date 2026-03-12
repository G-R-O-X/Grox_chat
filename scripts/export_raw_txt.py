import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "chatroom.db"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"


def _row_value(row: sqlite3.Row, key: str, default=None):
    return row[key] if key in row.keys() else default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one topic's raw discussion data into a readable text file."
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="Path to the SQLite database. Defaults to ./chatroom.db.",
    )
    parser.add_argument(
        "--topic-id",
        type=int,
        help="Topic ID to export. Defaults to the latest topic in the database.",
    )
    parser.add_argument(
        "--run-label",
        help="Label used in the header and default output filename, e.g. samplerun_007.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the exported text file. Defaults to ./outputs.",
    )
    parser.add_argument(
        "--output-file",
        help="Explicit output file path. Overrides --output-dir and --run-label naming.",
    )
    return parser.parse_args()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_topic_id(conn: sqlite3.Connection, topic_id: int | None) -> int:
    if topic_id is not None:
        row = conn.execute("SELECT id FROM Topic WHERE id = ?", (topic_id,)).fetchone()
        if row is None:
            raise SystemExit(f"Topic {topic_id} not found in {conn}.")
        return topic_id

    row = conn.execute("SELECT id FROM Topic ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        raise SystemExit("No topics found in database.")
    return int(row["id"])


def _load_topic_bundle(conn: sqlite3.Connection, topic_id: int) -> tuple[sqlite3.Row, sqlite3.Row | None, list[sqlite3.Row], list[sqlite3.Row], list[sqlite3.Row]]:
    topic = conn.execute("SELECT * FROM Topic WHERE id = ?", (topic_id,)).fetchone()
    if topic is None:
        raise SystemExit(f"Topic {topic_id} not found.")

    plan = conn.execute(
        "SELECT * FROM Plan WHERE topic_id = ? ORDER BY id DESC LIMIT 1",
        (topic_id,),
    ).fetchone()
    subtopics = conn.execute(
        "SELECT * FROM Subtopic WHERE topic_id = ? ORDER BY id ASC",
        (topic_id,),
    ).fetchall()
    messages = conn.execute(
        "SELECT * FROM Message WHERE topic_id = ? ORDER BY id ASC",
        (topic_id,),
    ).fetchall()
    facts = conn.execute(
        "SELECT * FROM Fact WHERE topic_id = ? ORDER BY id ASC",
        (topic_id,),
    ).fetchall()
    return topic, plan, subtopics, messages, facts


def _normalize_run_label(topic_id: int, run_label: str | None) -> str:
    if run_label:
        return run_label
    return f"topic_{topic_id:03d}"


def _resolve_output_path(output_dir: str, output_file: str | None, run_label: str) -> Path:
    if output_file:
        return Path(output_file)
    return Path(output_dir) / f"{run_label}_raw.txt"


def _render_plan(plan: sqlite3.Row | None) -> str:
    if plan is None:
        return "No plan stored.\n"

    try:
        plan_data = json.loads(plan["content"])
    except Exception:
        return f"{plan['content']}\n"

    rendered: list[str] = []
    for index, item in enumerate(plan_data, start=1):
        summary = item.get("summary", "").strip() or f"Subtopic {index}"
        detail = item.get("detail", "").strip()
        rendered.append(f"{index}. {summary}")
        if detail:
            rendered.append(f"   Detail: {detail}")
    return "\n".join(rendered) + "\n"


def _render_subtopics(subtopics: list[sqlite3.Row]) -> str:
    if not subtopics:
        return "No subtopics stored.\n"

    rendered: list[str] = []
    for subtopic in subtopics:
        rendered.append(
            f"- [{subtopic['id']}] {subtopic['summary']} | status={subtopic['status']} | created_at={subtopic['created_at']}"
        )
        detail = (_row_value(subtopic, "detail", "") or "").strip()
        if detail:
            rendered.append(f"  Detail: {detail}")
        conclusion = (_row_value(subtopic, "conclusion", "") or "").strip()
        if conclusion:
            rendered.append(f"  Conclusion: {conclusion}")
    return "\n".join(rendered) + "\n"


def _render_facts(facts: list[sqlite3.Row]) -> str:
    if not facts:
        return "No facts stored.\n"

    rendered: list[str] = []
    for fact in facts:
        rendered.append(
            f"[Fact {fact['id']}] source={fact['source']} stage={fact['fact_stage']} review={fact['review_status']} confidence={fact['confidence_score']}: {fact['content']}"
        )
    return "\n".join(rendered) + "\n"


def _render_messages(messages: list[sqlite3.Row], subtopics: list[sqlite3.Row]) -> str:
    if not messages:
        return "No messages stored.\n"

    subtopic_titles = {row["id"]: row["summary"] for row in subtopics}
    rendered: list[str] = []
    current_subtopic_id = None

    for msg in messages:
        if msg["subtopic_id"] != current_subtopic_id:
            current_subtopic_id = msg["subtopic_id"]
            title = subtopic_titles.get(current_subtopic_id, "General")
            rendered.append("")
            rendered.append(f"[SUBTOPIC {current_subtopic_id}: {title}]")
            rendered.append("=" * 72)

        rendered.append(
            " | ".join(
                [
                    f"ID={msg['id']}",
                    f"SENDER={msg['sender']}",
                    f"TYPE={msg['msg_type']}",
                    f"ROUND={_row_value(msg, 'round_number', '')}",
                    f"TURN_KIND={_row_value(msg, 'turn_kind', '') or ''}",
                    f"CREATED_AT={_row_value(msg, 'created_at', '')}",
                ]
            )
        )
        rendered.append("CONTENT:")
        rendered.append(msg["content"] or "")
        rendered.append("-" * 40)

    return "\n".join(rendered) + "\n"


def main() -> None:
    args = _parse_args()
    conn = _connect(args.db_path)
    try:
        topic_id = _resolve_topic_id(conn, args.topic_id)
        topic, plan, subtopics, messages, facts = _load_topic_bundle(conn, topic_id)
    finally:
        conn.close()

    run_label = _normalize_run_label(topic_id, args.run_label)
    output_path = _resolve_output_path(args.output_dir, args.output_file, run_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 72 + "\n")
        handle.write(f"GROX_CHAT EXPERIMENT RAW LOG: {run_label.upper()}\n")
        handle.write(f"EXPORT DATE (UTC): {export_date}\n")
        handle.write("=" * 72 + "\n\n")

        handle.write(f"TOPIC ID: {topic['id']}\n")
        handle.write(f"TOPIC SUMMARY: {topic['summary']}\n")
        handle.write(f"TOPIC DETAIL: {topic['detail']}\n")
        handle.write(f"STATUS: {topic['status']}\n")
        handle.write(f"CREATED_AT: {_row_value(topic, 'created_at', '')}\n")
        topic_conclusion = (_row_value(topic, "conclusion", "") or "").strip()
        if topic_conclusion:
            handle.write(f"CONCLUSION: {topic_conclusion}\n")
        handle.write("\n")

        handle.write("--- PLAN ---\n")
        handle.write(_render_plan(plan))
        handle.write("\n")

        handle.write("--- SUBTOPICS ---\n")
        handle.write(_render_subtopics(subtopics))
        handle.write("\n")

        handle.write("--- VERIFIED FACTS ---\n")
        handle.write(_render_facts(facts))
        handle.write("\n")

        handle.write("--- FULL CONVERSATION LOG ---\n")
        handle.write(_render_messages(messages, subtopics))

    print(f"Exported topic {topic_id} to {output_path}")


if __name__ == "__main__":
    main()
