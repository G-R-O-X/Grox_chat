import json
import os
from datetime import datetime, timezone

from aiohttp import web

from . import api
from .logging_utils import configure_logging


def _parse_plan_items(plan):
    if not plan:
        return []
    try:
        items = json.loads(plan.get("content") or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _phase_for_round(round_number):
    if not round_number or round_number <= 1:
        return "opening"
    if round_number == 2:
        return "evidence"
    return "debate"


def build_dashboard_snapshot(subtopic_id=None):
    topic = api.get_current_topic()
    if not topic:
        return {
            "topic": None,
            "plan": None,
            "subtopics": [],
            "current_subtopic": None,
            "messages": [],
            "facts": [],
            "fact_candidates": [],
            "status": {
                "db_path": api.get_db_path(),
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
                "current_round": None,
                "current_phase": None,
            },
        }

    topic_id = topic["id"]
    plan = api.get_active_plan(topic_id)
    subtopics = api.get_current_subtopics(topic_id)
    if subtopic_id:
        current_subtopic = api.get_subtopic(int(subtopic_id))
    else:
        current_subtopic = api.get_open_subtopic(topic_id) or api.get_latest_subtopic(topic_id)
    messages = api.get_messages(topic_id, subtopic_id=current_subtopic["id"] if current_subtopic else None, limit=120)
    facts = api.get_facts(topic_id, limit=80)
    fact_candidates = []
    if current_subtopic:
        fact_candidates = api.get_fact_candidates(topic_id, subtopic_id=current_subtopic["id"], limit=40)

    last_round = None
    for message in reversed(messages):
        if message.get("round_number") is not None:
            last_round = message["round_number"]
            break

    return {
        "topic": topic,
        "plan": {
            "id": plan["id"],
            "current_index": plan["current_index"],
            "items": _parse_plan_items(plan),
        } if plan else None,
        "subtopics": subtopics,
        "current_subtopic": current_subtopic,
        "messages": messages,
        "facts": facts,
        "fact_candidates": fact_candidates,
        "web_evidence": api.get_web_evidence_for_topic(topic_id),
        "status": {
            "db_path": api.get_db_path(),
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "current_round": last_round,
            "current_phase": _phase_for_round(last_round) if last_round is not None else None,
        },
    }


def render_dashboard_html():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GROX Chat Monitor</title>
  <style>
    :root {
      --bg: #f4efe5;
      --panel: #fffdf8;
      --ink: #1d1b18;
      --muted: #665f55;
      --line: #d9ccba;
      --accent: #0f5b4b;
      --warning: #8c4b16;
      --fact: #163e78;
      --pending: #7a3e00;
      --mono: "IBM Plex Mono", "SFMono-Regular", monospace;
      --serif: "Iowan Old Style", "Palatino Linotype", serif;
      --chat-sys-bg: #e6eef5;
      --chat-usr-bg: #eef5e6;
      --chat-dog-border: #cc444b;
      --chat-cat-border: #44cc77;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #efe7da 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: var(--serif);
    }
    .shell {
      max-width: 1600px;
      margin: 0 auto;
      padding: 20px;
    }
    .header {
      margin-bottom: 24px;
    }
    .eyebrow {
      font-family: var(--mono);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin-bottom: 8px;
    }
    h1 {
      margin: 0 0 12px 0;
      font-size: 28px;
      line-height: 1.2;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      font-family: var(--mono);
      font-size: 13px;
      color: var(--muted);
    }
    .nav-bar {
      margin-bottom: 20px;
      padding: 10px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      font-family: var(--mono);
      font-size: 14px;
    }
    .nav-bar select {
      padding: 4px;
      font-family: var(--mono);
      margin-left: 10px;
    }
    .grid {
      display: grid;
      grid-template-columns: 320px 1fr 320px;
      gap: 24px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: 0 2px 8px rgba(0,0,0,0.02);
      border-radius: 6px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 12px 16px;
      font-size: 13px;
      font-family: var(--mono);
      border-bottom: 1px solid var(--line);
      background: rgba(0,0,0,0.02);
      text-transform: uppercase;
    }
    .panel-body {
      padding: 14px 16px;
    }
    .stack {
      display: grid;
      gap: 12px;
    }
    /* Timeline Chat Styles */
    .timeline {
      display: flex;
      flex-direction: column;
      gap: 16px;
      align-items: flex-start;
    }
    .chat-bubble {
      padding: 12px;
      border-radius: 8px;
      width: 100%;
      max-width: 90%;
      border: 1px solid var(--line);
      background: #fff;
    }
    .chat-bubble.system {
      max-width: 100%;
      background: var(--chat-sys-bg);
      border-color: #c0d4e6;
    }
    .chat-bubble.special-dog { border: 2px solid var(--chat-dog-border); }
    .chat-bubble.special-cat { border: 2px solid var(--chat-cat-border); }
    
    .chat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
      font-family: var(--mono);
      font-size: 11px;
    }
    .chat-sender { font-weight: bold; text-transform: uppercase; }
    .chat-meta { color: var(--muted); }
    
    .chat-content {
      white-space: pre-wrap;
      line-height: 1.5;
      font-size: 14.5px;
    }
    
    details > summary {
      cursor: pointer;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--accent);
      font-weight: bold;
      margin-bottom: 8px;
    }
    
    /* Small items for side panels */
    .item {
      padding-bottom: 12px;
      border-bottom: 1px solid var(--line);
    }
    .item:last-child {
      border-bottom: 0;
      padding-bottom: 0;
    }
    .label {
      display: inline-block;
      padding: 2px 6px;
      border: 1px solid var(--line);
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      margin-right: 6px;
      margin-bottom: 6px;
    }
    .message-sender { font-weight: 700; margin-bottom: 4px; }
    .message-meta { font-family: var(--mono); color: var(--muted); font-size: 11px; margin-bottom: 6px; }
    .message-content, .fact-content, .candidate-content { white-space: pre-wrap; line-height: 1.45; font-size: 14px; }
    .fact-content { color: var(--fact); }
    .candidate-content { color: var(--pending); }
    .empty { color: var(--muted); font-style: italic; }
    
    @media (max-width: 1100px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div class="eyebrow">GROX Chat Live Monitor</div>
      <h1 id="topic-title">Loading...</h1>
      <div class="meta" id="topic-meta"></div>
    </div>
    
    <div class="nav-bar" id="nav-bar" style="display:none;">
      <label for="subtopic-select">View Subtopic:</label>
      <select id="subtopic-select" onchange="changeSubtopic(this.value)">
      </select>
    </div>

    <div class="grid">
      <section class="panel">
        <h2>Plan & Subtopics</h2>
        <div class="panel-body stack" id="plan-panel"></div>
      </section>
      <section class="panel">
        <h2>Timeline</h2>
        <div class="panel-body timeline" id="messages-panel"></div>
      </section>
      <section class="panel">
        <h2>Knowledge Base</h2>
        <div class="panel-body">
          <div class="stack" id="facts-panel"></div>
          <hr style="border:none;border-top:1px solid var(--line);margin:16px 0;">
          <div class="stack" id="candidates-panel"></div>
        </div>
      </section>
    </div>
  </div>
  <script>
    let currentSubtopicId = new URLSearchParams(window.location.search).get("subtopic_id");

    function changeSubtopic(id) {
        if(id) {
            window.location.href = "?subtopic_id=" + id;
        } else {
            window.location.href = "/";
        }
    }

    function esc(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function renderEmpty(node, text) {
      node.innerHTML = '<div class="empty">' + esc(text) + '</div>';
    }

    function renderTopic(snapshot) {
      const title = document.getElementById('topic-title');
      const meta = document.getElementById('topic-meta');
      const navBar = document.getElementById('nav-bar');
      const subSelect = document.getElementById('subtopic-select');
      
      if (!snapshot.topic) {
        title.textContent = 'No active topic';
        meta.innerHTML = '<span>' + esc('No data in database') + '</span>';
        return;
      }
      title.textContent = snapshot.topic.summary;
      const bits = [
        'status=' + snapshot.topic.status,
        'subtopic=' + (snapshot.current_subtopic ? snapshot.current_subtopic.summary : 'none'),
        'round=' + (snapshot.status.current_round ?? '-'),
        'phase=' + (snapshot.status.current_phase ?? '-'),
        'refreshed=' + snapshot.status.refreshed_at,
      ];
      meta.innerHTML = bits.map(bit => '<span>' + esc(bit) + '</span>').join('');
      
      if(snapshot.subtopics && snapshot.subtopics.length > 0) {
          navBar.style.display = "block";
          let opts = "<option value=''>-- Latest --</option>";
          snapshot.subtopics.forEach(st => {
              let selected = (currentSubtopicId && st.id == currentSubtopicId) || (!currentSubtopicId && snapshot.current_subtopic && st.id == snapshot.current_subtopic.id) ? "selected" : "";
              opts += "<option value='" + st.id + "' " + selected + ">#" + st.id + " " + esc(st.summary) + "</option>";
          });
          subSelect.innerHTML = opts;
      }
    }

    function renderPlan(snapshot) {
      const node = document.getElementById('plan-panel');
      if (!snapshot.topic) {
        renderEmpty(node, 'No topic loaded.');
        return;
      }
      const items = [];
      if (snapshot.plan) {
        items.push('<div class="item"><div class="message-meta">' + esc('plan_id=' + snapshot.plan.id + ' current_index=' + snapshot.plan.current_index) + '</div></div>');
        snapshot.plan.items.forEach((item, index) => {
          items.push(
            '<div class="item">' +
            '<div><span class="label">' + esc('plan ' + (index + 1)) + '</span></div>' +
            '<div class="message-sender">' + esc(item.summary) + '</div>' +
            '<div class="message-content">' + esc(item.detail) + '</div>' +
            '</div>'
          );
        });
      }
      snapshot.subtopics.forEach((subtopic) => {
        items.push(
          '<div class="item">' +
          '<div><span class="label">' + esc('subtopic #' + subtopic.id) + '</span><span class="label">' + esc(subtopic.status) + '</span></div>' +
          '<div class="message-sender">' + esc(subtopic.summary) + '</div>' +
          '<div class="message-content">' + esc(subtopic.conclusion || subtopic.detail) + '</div>' +
          '</div>'
        );
      });
      if (!items.length) {
        renderEmpty(node, 'Plan not generated yet.');
        return;
      }
      node.innerHTML = items.join('');
    }

    function renderMessages(snapshot) {
      const node = document.getElementById('messages-panel');
      if (!snapshot.messages.length) {
        renderEmpty(node, 'No messages yet.');
        return;
      }
      
      const adminRoles = ["skynet", "librarian", "writer", "tron"];
      
      node.innerHTML = snapshot.messages.map((message) => {
        let classes = "chat-bubble";
        if(adminRoles.includes(message.sender.toLowerCase())) {
            classes += " system";
        } else if (message.sender.toLowerCase() === "dog") {
            classes += " special-dog";
        } else if (message.sender.toLowerCase() === "cat") {
            classes += " special-cat";
        }
        
        const isLongText = message.content.length > 400 || message.msg_type === "summary" || message.turn_kind === "librarian_audit";
        
        let contentHtml = "";
        if(isLongText) {
            let previewText = esc(message.content.substring(0, 150)) + "...";
            if (message.msg_type === "summary") previewText = "Summary Content";
            if (message.turn_kind === "librarian_audit") previewText = "Librarian Audit Log";
            
            contentHtml = '<details id="msg-det-' + message.id + '"><summary>Expand: ' + previewText + '</summary><div class="chat-content">' + esc(message.content) + '</div></details>';
        } else {
            contentHtml = '<div class="chat-content">' + esc(message.content) + '</div>';
        }

        const metaLabels = [];
        if (message.round_number !== null && message.round_number !== undefined) {
          metaLabels.push('R' + message.round_number);
        }
        if (message.turn_kind) metaLabels.push(message.turn_kind);
        if (message.msg_type !== "standard") metaLabels.push(message.msg_type);

        return (
          '<div class="' + classes + '">' +
          '<div class="chat-header">' +
          '<span class="chat-sender">' + esc(message.sender) + '</span>' +
          '<span class="chat-meta">' + esc(metaLabels.join(' | ')) + ' (id=' + message.id + ')</span>' +
          '</div>' +
          contentHtml +
          '</div>'
        );
      }).join('');
    }

    function renderFacts(snapshot) {
      const factsNode = document.getElementById('facts-panel');
      const candidatesNode = document.getElementById('candidates-panel');
      
      let html = '';
      if (!snapshot.facts || !snapshot.facts.length) {
        html += '<div class="empty">No accepted facts yet.</div>';
      } else {
        html += '<h3>[F] Facts</h3>' + snapshot.facts.map((fact) => (
          '<div class="item">' +
          '<div><span class="label">[F' + fact.id + ']</span><span class="label">' + esc(fact.review_status || 'accepted') + '</span></div>' +
          '<div class="fact-content">' + esc(fact.content) + '</div>' +
          '</div>'
        )).join('');
      }
      
      if (snapshot.web_evidence && snapshot.web_evidence.length) {
        html += '<hr style="border:none;border-top:1px solid var(--line);margin:16px 0;"><h3>[W] Web Evidence</h3>';
        html += '<details id="web-evidence-det"><summary>View ' + snapshot.web_evidence.length + ' stored web sources</summary>';
        html += snapshot.web_evidence.map((we) => (
          '<div class="item" style="margin-top:12px;">' +
          '<div><span class="label">[W' + we.id + ']</span><a href="' + esc(we.url) + '" target="_blank" class="message-meta">' + esc(we.source_domain) + '</a></div>' +
          '<div class="message-sender">' + esc(we.title) + '</div>' +
          '<div class="candidate-content">' + esc(we.snippet) + '</div>' +
          '</div>'
        )).join('');
        html += '</details>';
      }
      factsNode.innerHTML = html;
      
      if (!snapshot.fact_candidates || !snapshot.fact_candidates.length) {
        renderEmpty(candidatesNode, 'No fact candidates in this subtopic yet.');
      } else {
        candidatesNode.innerHTML = '<h3>Fact Candidates</h3>' + snapshot.fact_candidates.map((candidate) => {
          let extraHtml = '';
          if (candidate.status !== 'pending' && candidate.review_note) {
              extraHtml = '<details id="cand-det-' + candidate.id + '"><summary>View Audit Reason</summary><div class="message-meta">' + esc(candidate.review_note) + '</div></details>';
          }
          return '<div class="item">' +
          '<div><span class="label">cand #' + candidate.id + '</span><span class="label">' + esc(candidate.status) + '</span></div>' +
          '<div class="candidate-content">' + esc(candidate.candidate_text) + '</div>' +
          extraHtml +
          '</div>';
        }).join('');
      }
    }

    async function refresh() {
      let url = '/api/dashboard';
      if(currentSubtopicId) url += '?subtopic_id=' + currentSubtopicId;
      const response = await fetch(url, { cache: 'no-store' });
      const snapshot = await response.json();
      
      // Save state of <details> tags
      const detailsState = {};
      document.querySelectorAll('details').forEach(el => {
         if(el.id) detailsState[el.id] = el.open;
      });

      renderTopic(snapshot);
      renderPlan(snapshot);
      renderMessages(snapshot);
      renderFacts(snapshot);
      
      // Restore state
      document.querySelectorAll('details').forEach(el => {
         if(el.id && detailsState[el.id]) el.open = true;
      });
      
      // If the topic is closed, we can stop hammering the server
      if (snapshot.topic && snapshot.topic.status === 'Closed') {
          if (window.refreshInterval) {
              clearInterval(window.refreshInterval);
          }
      }
    }

    refresh();
    window.refreshInterval = setInterval(refresh, 2000);
  </script>
</body>
</html>"""


async def index(request):
    return web.Response(text=render_dashboard_html(), content_type="text/html")


async def dashboard(request):
    subtopic_id = request.query.get('subtopic_id')
    return web.json_response(build_dashboard_snapshot(subtopic_id=subtopic_id))


async def health(request):
    snapshot = build_dashboard_snapshot()
    return web.json_response(
        {
            "ok": True,
            "db_path": snapshot["status"]["db_path"],
            "topic_id": snapshot["topic"]["id"] if snapshot["topic"] else None,
            "current_subtopic_id": snapshot["current_subtopic"]["id"] if snapshot["current_subtopic"] else None,
        }
    )


def create_app():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/dashboard", dashboard)
    app.router.add_get("/api/health", health)
    return app


def main():
    configure_logging()
    host = os.environ.get("GROX_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("GROX_WEB_PORT", "8080"))
    web.run_app(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
