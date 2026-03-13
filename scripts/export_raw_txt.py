import os
import sys
import sqlite3
from typing import Dict, Any

def export_topic_to_txt(db_path: str, topic_id: int, output_dir: str):
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"topic_{topic_id}_export.md")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get Topic
    topic = cursor.execute("SELECT * FROM Topic WHERE id = ?", (topic_id,)).fetchone()
    if not topic:
        print(f"Error: Topic {topic_id} not found.")
        conn.close()
        return

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"# Topic {topic_id}: {topic['summary']}\n\n")
        f.write(f"**Detail:** {topic['detail']}\n\n")
        f.write(f"**Status:** {topic['status']}\n\n")
        f.write("---\n\n")

        # Get Subtopics
        subtopics = cursor.execute("SELECT * FROM Subtopic WHERE topic_id = ? ORDER BY id", (topic_id,)).fetchall()
        
        for st in subtopics:
            f.write(f"## Subtopic {st['id']}: {st['summary']}\n\n")
            f.write(f"> {st['detail']}\n\n")
            
            # Get Messages
            messages = cursor.execute("SELECT * FROM Message WHERE subtopic_id = ? ORDER BY id", (st['id'],)).fetchall()
            
            # Get VoteRecords
            votes = cursor.execute("SELECT * FROM VoteRecord WHERE subtopic_id = ? ORDER BY id", (st['id'],)).fetchall()
            vote_dict = {}
            for v in votes:
                round_num = v['round_number']
                if round_num not in vote_dict:
                    vote_dict[round_num] = []
                vote_dict[round_num].append(v)
            
            current_round = None
            
            for msg in messages:
                msg_round = msg['round_number']
                
                # Check if we transitioned to a new round
                if msg_round != current_round and msg_round is not None:
                    # Print votes for the *previous* round before starting the new one, if any
                    if current_round in vote_dict:
                        f.write(f"### [Governance] Votes for Round {current_round}\n\n")
                        for v in vote_dict[current_round]:
                            f.write(f"- **{v['voter']}** -> `{v['decision']}`: {v['reason']}\n")
                        f.write("\n")
                        
                    current_round = msg_round
                    f.write(f"\n==================== ROUND {current_round} ====================\n\n")

                sender = msg['sender'].upper()
                content = msg['content']
                
                if msg['msg_type'] == 'summary':
                    f.write(f"### 🤖 SKYNET SUMMARY (Round {msg_round})\n\n")
                    f.write(f"{content}\n\n")
                elif msg['msg_type'] == 'governance' or msg['sender'] in ('tron', 'librarian'):
                    f.write(f"### 🛡️ {sender} AUDIT\n\n")
                    f.write(f"{content}\n\n")
                elif msg['sender'] == 'writer':
                    f.write(f"### ✍️ WRITER CRITIQUE\n\n")
                    f.write(f"{content}\n\n")
                else:
                    f.write(f"**[{sender}]**:\n{content}\n\n")
            
            # Print any trailing votes for the final round
            if current_round in vote_dict:
                f.write(f"### [Governance] Votes for Round {current_round}\n\n")
                for v in vote_dict[current_round]:
                    f.write(f"- **{v['voter']}** -> `{v['decision']}`: {v['reason']}\n")
                f.write("\n")
                
            f.write("---\n\n")
            
        # Facts
        f.write("## 📚 Accepted Facts\n\n")
        facts = cursor.execute("SELECT * FROM Fact WHERE topic_id = ?", (topic_id,)).fetchall()
        for fact in facts:
            f.write(f"- **[F{fact['id']}]**: {fact['content']}\n")
            
        f.write("\n## 🌐 Web Evidence Context\n\n")
        webs = cursor.execute("SELECT * FROM WebEvidence WHERE origin_topic_id = ?", (topic_id,)).fetchall()
        for w in webs:
            f.write(f"- **[W{w['id']}]** [{w['source_domain']}]: {w['snippet']}\n")

    print(f"Successfully exported Topic {topic_id} to {out_file}")
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_raw_txt.py <topic_id> [db_path]")
        sys.exit(1)
    
    t_id = int(sys.argv[1])
    db = sys.argv[2] if len(sys.argv) > 2 else "chatroom.db"
    export_topic_to_txt(db, t_id, "outputs")
