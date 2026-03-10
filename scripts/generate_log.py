import argparse
import json
import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "chatroom.db"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "samplerun_001.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a topic transcript from the local GROX Chat SQLite database.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to the SQLite database.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to the output log file.")
    parser.add_argument("--topic-id", type=int, default=1, help="Topic ID to export.")
    return parser.parse_args()


def get_data(db_path: Path, topic_id: int):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    topic = conn.execute("SELECT * FROM Topic WHERE id = ?", (topic_id,)).fetchone()
    plan = conn.execute("SELECT * FROM Plan WHERE topic_id = ? ORDER BY id DESC LIMIT 1", (topic_id,)).fetchone()
    subtopics = conn.execute("SELECT * FROM Subtopic WHERE topic_id = ? ORDER BY id ASC", (topic_id,)).fetchall()
    messages = conn.execute(
        "SELECT id, subtopic_id, sender, content, msg_type FROM Message WHERE topic_id = ? ORDER BY id ASC",
        (topic_id,),
    ).fetchall()

    conn.close()
    return topic, plan, subtopics, messages


def render_log(topic, plan, subtopics, messages) -> str:
    lines: list[str] = []
    lines.append(f"=== GROX Chat Run Log / Topic {topic['id']} ===")
    lines.append("")
    lines.append(f"Topic: {topic['summary']}")
    lines.append(f"Detail: {topic['detail']}")
    lines.append("")
    lines.append("=== Plan ===")
    if plan is None:
        lines.append("(no plan found)")
    else:
        try:
            plan_items = json.loads(plan["content"])
            for i, item in enumerate(plan_items, start=1):
                lines.append(f"{i}. {item['summary']}: {item['detail']}")
        except Exception:
            lines.append(plan["content"])
    lines.append("")

    for sub in subtopics:
        lines.append(f"--- Subtopic {sub['id']}: {sub['summary']} ---")
        lines.append(f"Detail: {sub['detail']}")
        lines.append(f"Status: {sub['status']}")
        lines.append("")
        sub_msgs = [m for m in messages if m["subtopic_id"] == sub["id"]]
        for m in sub_msgs:
            lines.append(f"[{m['sender']}] ({m['msg_type']}):")
            lines.append(m["content"])
            lines.append("-" * 20)
        lines.append("")

    topic_level_messages = [m for m in messages if m["subtopic_id"] is None]
    if topic_level_messages:
        lines.append("=== Topic-level Messages ===")
        for m in topic_level_messages:
            lines.append(f"[{m['sender']}] ({m['msg_type']}):")
            lines.append(m["content"])
            lines.append("-" * 20)
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    topic, plan, subtopics, messages = get_data(db_path, args.topic_id)
    if topic is None:
        raise SystemExit(f"Topic {args.topic_id} not found in {db_path}")

    with output_path.open("w", encoding="utf-8") as f:
        f.write(render_log(topic, plan, subtopics, messages))


if __name__ == "__main__":
    main()
