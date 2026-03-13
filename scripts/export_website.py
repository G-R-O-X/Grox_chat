import os
import sys
import json
import sqlite3

def export_website(db_path: str, topic_id: int, output_dir: str):
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    # To use grox_chat.web, we need to make sure the app context is available
    # Since build_dashboard_snapshot relies on api.py which reads DB_PATH
    os.environ["GROX_DB_PATH"] = db_path
    
    # Pre-add to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    from src.grox_chat import api
    from src.grox_chat.web import build_dashboard_snapshot, render_dashboard_html
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"topic_{topic_id}_website.html")
    
    topic = api.get_topic(topic_id)
    if not topic:
        print(f"Error: Topic {topic_id} not found in DB.")
        return
        
    subtopics = api.get_current_subtopics(topic_id)
    
    static_data = {
        "subtopics": subtopics,
        "subtopic_data": {}
    }
    
    # Generate snapshot for the topic overall (no specific subtopic)
    print("Generating default snapshot...")
    static_data["subtopic_data"]["default"] = build_dashboard_snapshot()
    
    # Generate snapshots for each specific subtopic
    for st in subtopics:
        print(f"Generating snapshot for Subtopic {st['id']}...")
        static_data["subtopic_data"][st["id"]] = build_dashboard_snapshot(subtopic_id=st["id"])
        
    json_str = json.dumps(static_data)
    
    # Get the HTML template
    html_template = render_dashboard_html()
    
    # Inject the payload right before </head>
    injection = f"\n<script>window.GROX_STATIC_DATA = {json_str};</script>\n"
    html_out = html_template.replace("</head>", injection + "</head>")
    
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(html_out)
        
    print(f"\n✅ Successfully exported Topic {topic_id} as an interactive offline website to: {out_file}")
    print("You can now open this file directly in any web browser without running the Python server.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/export_website.py <topic_id> [db_path]")
        sys.exit(1)
    
    t_id = int(sys.argv[1])
    db = sys.argv[2] if len(sys.argv) > 2 else "chatroom.db"
    export_website(db, t_id, "outputs")
