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


def build_dashboard_snapshot():
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
    current_subtopic = api.get_open_subtopic(topic_id) or api.get_latest_subtopic(topic_id)
    messages = api.get_messages(topic_id, subtopic_id=current_subtopic["id"] if current_subtopic else None, limit=120)
    facts = api.get_facts(topic_id, limit=80)
    fact_candidates = []
    if current_subtopic:
        fact_candidates = api.get_fact_candidates(topic_id, subtopic_id=current_subtopic["id"], status="pending", limit=40)

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
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 16px 18px;
      margin-bottom: 18px;
      display: grid;
      gap: 10px;
    }
    .eyebrow {
      font-family: var(--mono);
      font-size: 12px;
      letter-spacing: 0.08em;
      color: var(--muted);
      text-transform: uppercase;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr) 320px;
      gap: 18px;
    }
    .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      min-height: 200px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      font-family: var(--mono);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .panel-body {
      padding: 14px 16px;
    }
    .stack {
      display: grid;
      gap: 12px;
    }
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
    .message-sender {
      font-weight: 700;
      margin-bottom: 4px;
    }
    .message-meta {
      font-family: var(--mono);
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 6px;
    }
    .message-content, .fact-content, .candidate-content {
      white-space: pre-wrap;
      line-height: 1.45;
      font-size: 15px;
    }
    .fact-content { color: var(--fact); }
    .candidate-content { color: var(--pending); }
    .empty {
      color: var(--muted);
      font-style: italic;
    }
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
    <div class="grid">
      <section class="panel">
        <h2>Plan & Subtopics</h2>
        <div class="panel-body stack" id="plan-panel"></div>
      </section>
      <section class="panel">
        <h2>Timeline</h2>
        <div class="panel-body stack" id="messages-panel"></div>
      </section>
      <section class="panel">
        <h2>Facts & Pending Reviews</h2>
        <div class="panel-body">
          <div class="stack" id="facts-panel"></div>
          <hr style="border:none;border-top:1px solid var(--line);margin:16px 0;">
          <div class="stack" id="candidates-panel"></div>
        </div>
      </section>
    </div>
  </div>
  <script>
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
      node.innerHTML = snapshot.messages.map((message) => {
        const labels = [
          '<span class="label">' + esc(message.sender) + '</span>',
          '<span class="label">' + esc(message.msg_type) + '</span>'
        ];
        if (message.round_number !== null && message.round_number !== undefined) {
          labels.push('<span class="label">' + esc('round ' + message.round_number) + '</span>');
        }
        if (message.turn_kind) {
          labels.push('<span class="label">' + esc(message.turn_kind) + '</span>');
        }
        return (
          '<div class="item">' +
          '<div>' + labels.join('') + '</div>' +
          '<div class="message-meta">' + esc('id=' + message.id + (message.confidence_score != null ? ' confidence=' + message.confidence_score : '')) + '</div>' +
          '<div class="message-content">' + esc(message.content) + '</div>' +
          '</div>'
        );
      }).join('');
    }

    function renderFacts(snapshot) {
      const factsNode = document.getElementById('facts-panel');
      const candidatesNode = document.getElementById('candidates-panel');
      if (!snapshot.facts.length) {
        renderEmpty(factsNode, 'No accepted facts yet.');
      } else {
        factsNode.innerHTML = snapshot.facts.map((fact) => (
          '<div class="item">' +
          '<div><span class="label">' + esc('fact #' + fact.id) + '</span><span class="label">' + esc(fact.review_status || 'accepted') + '</span></div>' +
          '<div class="fact-content">' + esc(fact.content) + '</div>' +
          '</div>'
        )).join('');
      }
      if (!snapshot.fact_candidates.length) {
        renderEmpty(candidatesNode, 'No pending fact reviews.');
      } else {
        candidatesNode.innerHTML = snapshot.fact_candidates.map((candidate) => (
          '<div class="item">' +
          '<div><span class="label">' + esc('candidate #' + candidate.id) + '</span><span class="label">' + esc(candidate.status) + '</span></div>' +
          '<div class="candidate-content">' + esc(candidate.candidate_text) + '</div>' +
          '</div>'
        )).join('');
      }
    }

    async function refresh() {
      const response = await fetch('/api/dashboard', { cache: 'no-store' });
      const snapshot = await response.json();
      renderTopic(snapshot);
      renderPlan(snapshot);
      renderMessages(snapshot);
      renderFacts(snapshot);
    }

    refresh();
    setInterval(refresh, 1500);
  </script>
</body>
</html>"""


async def index(request):
    return web.Response(text=render_dashboard_html(), content_type="text/html")


async def dashboard(request):
    return web.json_response(build_dashboard_snapshot())


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
