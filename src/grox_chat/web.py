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
    claims = api.get_claims(topic_id, limit=80)
    fact_candidates = []
    claim_candidates = []
    votes = []
    if current_subtopic:
        fact_candidates = api.get_fact_candidates(topic_id, subtopic_id=current_subtopic["id"], limit=40)
        claim_candidates = api.get_claim_candidates(topic_id, subtopic_id=current_subtopic["id"], limit=40)
        votes = api.get_vote_records(topic_id, subtopic_id=current_subtopic["id"], limit=100)

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
        "claims": claims,
        "fact_candidates": fact_candidates,
        "claim_candidates": claim_candidates,
        "web_evidence": api.get_web_evidence_for_topic(topic_id),
        "status": {
            "db_path": api.get_db_path(),
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "current_round": last_round,
            "current_phase": _phase_for_round(last_round) if last_round is not None else None,
        },
    }


def render_dashboard_html():
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GROX Chat Monitor</title>
  <link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=JetBrains+Mono:wght@400;700&family=Work+Sans:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #141517;
      --panel: #232529;
      --ink: #e0e0e0;
      --muted: #8b8f98;
      --line: #3a3f4b;
      --accent: #00e5bf;
      --warning: #ffb400;
      --fact: #4da6ff;
      --pending: #e59866;
      --mono: "JetBrains Mono", monospace;
      --sans: "Work Sans", sans-serif;
      --display: "Archivo Black", sans-serif;
      --chat-sys-bg: #1b2633;
      --chat-sys-border: #2c4463;
      --chat-usr-bg: #2a2d32;
      --chat-usr-border: #3a3f4b;
      --chat-dog-border: #d33c46;
      --chat-cat-border: #3cb878;
    }
    
    * { box-sizing: border-box; }
    
    body {
      margin: 0;
      background-color: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden; /* Prevent body scroll */
    }
    
    .header {
      padding: 16px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
    }
    
    .header-left h1 {
      margin: 0 0 4px 0;
      font-family: var(--display);
      font-size: 20px;
      letter-spacing: -0.02em;
      color: #fff;
    }
    
    .eyebrow {
      font-family: var(--mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--accent);
    }
    
    .meta {
      display: flex;
      gap: 16px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--muted);
    }
    
    .meta span {
      background: #1a1c20;
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 4px;
    }

    .nav-bar select {
      padding: 6px;
      font-family: var(--mono);
      font-size: 12px;
      background: var(--bg);
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 4px;
      outline: none;
    }
    
    .main-container {
      display: grid;
      grid-template-columns: 320px 1fr 380px;
      gap: 1px;
      flex: 1;
      background: var(--line); /* acts as borders between columns */
      overflow: hidden;
    }
    
    .column {
      background: var(--bg);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      height: 100%;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--muted); }
    
    .panel-header {
      position: sticky;
      top: 0;
      background: var(--panel);
      padding: 12px 16px;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      border-bottom: 1px solid var(--line);
      z-index: 10;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .panel-body {
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    
    /* Chat Timeline Styles */
    .timeline {
      padding: 24px;
      gap: 20px;
      align-items: flex-start;
    }
    
    .chat-bubble {
      padding: 14px;
      border-radius: 6px;
      width: 100%;
      max-width: 90%;
      background: var(--chat-usr-bg);
      border: 1px solid var(--chat-usr-border);
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .chat-bubble.system {
      max-width: 100%;
      background: var(--chat-sys-bg);
      border-color: var(--chat-sys-border);
    }
    
    .chat-bubble.system .chat-sender { color: var(--accent); }
    .chat-bubble.special-dog { border-left: 4px solid var(--chat-dog-border); }
    .chat-bubble.special-cat { border-left: 4px solid var(--chat-cat-border); }
    
    .chat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      font-family: var(--mono);
      font-size: 11px;
      border-bottom: 1px dashed var(--line);
      padding-bottom: 6px;
    }
    
    .chat-sender { 
      font-weight: 700; 
      text-transform: uppercase; 
      color: #fff;
    }
    .chat-meta { color: var(--muted); }
    
    .chat-content {
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 14px;
    }
    
    details {
      background: rgba(0,0,0,0.2);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 8px;
    }
    
    details > summary {
      cursor: pointer;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--accent);
      font-weight: bold;
      outline: none;
    }
    
    details[open] summary {
      margin-bottom: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }
    
    /* Cards for side panels */
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 12px;
    }
    
    .label {
      display: inline-block;
      padding: 2px 6px;
      background: #1a1c20;
      border: 1px solid var(--line);
      font-family: var(--mono);
      font-size: 10px;
      color: var(--muted);
      margin-right: 6px;
      margin-bottom: 6px;
      border-radius: 2px;
    }
    
    .card-title { font-weight: 600; margin-bottom: 6px; font-size: 13px; color: #fff;}
    .card-content { white-space: pre-wrap; line-height: 1.5; font-size: 13px; color: #ccc;}
    .fact-content { color: var(--fact); font-family: var(--mono); font-size: 12px;}
    .candidate-content { color: var(--pending); font-size: 13px;}
    .empty { color: var(--muted); font-style: italic; font-size: 13px; text-align: center; padding: 20px;}
    
    /* Tabs for KB */
    .tabs {
      display: flex;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 45px;
      background: var(--bg);
      z-index: 9;
    }
    .tab {
      flex: 1;
      text-align: center;
      padding: 10px;
      font-family: var(--mono);
      font-size: 11px;
      cursor: pointer;
      color: var(--muted);
      border-bottom: 2px solid transparent;
    }
    .tab:hover { color: #fff; }
    .tab.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
    }
    .tab-content { display: none; }
    .tab-content.active { display: flex; flex-direction: column; gap: 12px; padding: 16px; }

    /* Tooltip styles */
    .citation {
      color: var(--accent);
      text-decoration: underline dotted;
      cursor: help;
      font-weight: bold;
      position: relative;
    }
    #global-tooltip {
      position: fixed;
      display: none;
      background: var(--panel);
      border: 1px solid var(--accent);
      padding: 12px;
      border-radius: 4px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      z-index: 10000;
      max-width: 400px;
      font-size: 13px;
      line-height: 1.5;
      pointer-events: none;
      color: #fff;
    }
    #global-tooltip .tt-label {
      font-family: var(--mono);
      font-size: 10px;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 4px;
      display: block;
      border-bottom: 1px solid var(--line);
      padding-bottom: 4px;
    }
  </style>
</head>
<body>
  <div id="global-tooltip"></div>
  <div class="header">
    <div class="header-left">
      <div class="eyebrow">GROX Chat Monitor</div>
      <h1 id="topic-title">Loading...</h1>
    </div>
    <div class="header-right" style="display:flex; align-items:center; gap:20px;">
      <div class="nav-bar" id="nav-bar" style="display:none;">
        <select id="subtopic-select" onchange="changeSubtopic(this.value)"></select>
      </div>
      <div class="meta" id="topic-meta"></div>
    </div>
  </div>
  
  <div class="main-container">
    <!-- Left Column: Plan & Details -->
    <div class="column">
      <div class="panel-header">Topic Details</div>
      <div class="panel-body">
        <div id="topic-detail" class="card-content" style="margin-bottom:20px;"></div>
        <div id="plan-panel" style="display:flex;flex-direction:column;gap:12px;"></div>
      </div>
    </div>
    
    <!-- Middle Column: Timeline -->
    <div class="column" id="scroll-timeline">
      <div class="panel-header">
        <span>Timeline</span>
        <span id="timeline-stats" style="color:var(--muted);font-weight:normal;"></span>
      </div>
      <div class="panel-body timeline" id="messages-panel"></div>
    </div>
    
    <!-- Right Column: Knowledge Base -->
    <div class="column">
      <div class="panel-header">Knowledge Base</div>
      <div class="tabs">
        <div class="tab active" onclick="switchTab('facts')">Facts</div>
        <div class="tab" onclick="switchTab('claims')">Claims</div>
        <div class="tab" onclick="switchTab('cands')">Pending</div>
        <div class="tab" onclick="switchTab('web')">Web</div>
      </div>
      
      <div id="tab-facts" class="tab-content active"></div>
      <div id="tab-claims" class="tab-content"></div>
      <div id="tab-cands" class="tab-content"></div>
      <div id="tab-web" class="tab-content"></div>
    </div>
  </div>

  <script>
    const KNOWLEDGE_MAP = {};

    function updateKnowledgeMap(snapshot) {
      if (snapshot.facts) snapshot.facts.forEach(f => KNOWLEDGE_MAP['F' + f.id] = { type: 'Fact', content: f.content });
      if (snapshot.claims) snapshot.claims.forEach(c => KNOWLEDGE_MAP['C' + c.id] = { type: 'Claim', content: c.content });
      if (snapshot.web_evidence) snapshot.web_evidence.forEach(w => KNOWLEDGE_MAP['W' + w.id] = { type: 'Web Source', content: w.title + ': ' + w.snippet });
    }

    function linkCitations(text) {
      return esc(text).replace(/\[([FWC])(\d+)\]/g, (match, type, id) => {
        const key = type + id;
        return '<span class="citation" onmouseover="showTooltip(event, \'' + key + '\')" onmouseout="hideTooltip()">[' + key + ']</span>';
      });
    }

    function showTooltip(e, key) {
      const tt = document.getElementById('global-tooltip');
      const data = KNOWLEDGE_MAP[key];
      if (!data) return;
      
      tt.innerHTML = '<span class="tt-label">' + esc(data.type) + ' [' + key + ']</span>' + esc(data.content);
      tt.style.display = 'block';
      
      // Positioning
      const x = e.clientX + 15;
      const y = e.clientY + 15;
      tt.style.left = x + 'px';
      tt.style.top = y + 'px';
      
      // Flip if overflow
      const rect = tt.getBoundingClientRect();
      if (rect.right > window.innerWidth) tt.style.left = (e.clientX - rect.width - 15) + 'px';
      if (rect.bottom > window.innerHeight) tt.style.top = (e.clientY - rect.height - 15) + 'px';
    }

    function hideTooltip() {
      document.getElementById('global-tooltip').style.display = 'none';
    }

    let currentSubtopicId = new URLSearchParams(window.location.search).get("subtopic_id");

    function changeSubtopic(id) {
        if (window.GROX_STATIC_DATA) {
            currentSubtopicId = id;
            refresh();
        } else {
            if(id) window.location.href = "?subtopic_id=" + id;
            else window.location.href = "/";
        }
    }

    function switchTab(tabName) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById('tab-' + tabName).classList.add('active');
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
      const detail = document.getElementById('topic-detail');
      const meta = document.getElementById('topic-meta');
      const navBar = document.getElementById('nav-bar');
      const subSelect = document.getElementById('subtopic-select');
      
      if (!snapshot.topic) {
        title.textContent = 'No active topic';
        meta.innerHTML = '<span>No data</span>';
        return;
      }
      
      // Truncate title if too long
      let displayTitle = snapshot.topic.summary;
      if (displayTitle.length > 70) displayTitle = displayTitle.substring(0, 70) + "...";
      title.textContent = displayTitle;
      title.title = snapshot.topic.summary; // full text on hover
      
      detail.textContent = snapshot.topic.detail;
      
      const bits = [
        'STATUS:' + snapshot.topic.status,
        'ROUND:' + (snapshot.status.current_round ?? '-'),
        'PHASE:' + (snapshot.status.current_phase ?? '-')
      ];
      meta.innerHTML = bits.map(bit => '<span>' + esc(bit) + '</span>').join('');
      
      if(snapshot.subtopics && snapshot.subtopics.length > 0) {
          navBar.style.display = "block";
          let opts = "<option value=''>-- Latest Subtopic --</option>";
          snapshot.subtopics.forEach(st => {
              let selected = (currentSubtopicId && st.id == currentSubtopicId) || (!currentSubtopicId && snapshot.current_subtopic && st.id == snapshot.current_subtopic.id) ? "selected" : "";
              opts += "<option value='" + st.id + "' " + selected + ">#" + st.id + " " + esc(st.summary.substring(0, 30)) + "...</option>";
          });
          subSelect.innerHTML = opts;
      }
    }

    function renderPlan(snapshot) {
      const node = document.getElementById('plan-panel');
      if (!snapshot.topic) return;
      
      const items = [];
      snapshot.subtopics.forEach((st) => {
        let statusColor = st.status === 'Open' ? 'var(--accent)' : (st.status === 'Closed' ? 'var(--muted)' : 'var(--warning)');
        items.push(
          '<div class="card" style="border-left: 3px solid ' + statusColor + '">' +
          '<div><span class="label">ST #' + st.id + '</span><span class="label" style="color:' + statusColor + '">' + esc(st.status) + '</span></div>' +
          '<div class="card-title">' + esc(st.summary) + '</div>' +
          '<div class="card-content">' + esc(st.conclusion || st.detail) + '</div>' +
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
      const statsNode = document.getElementById('timeline-stats');
      
      if (!snapshot.messages.length && (!snapshot.votes || !snapshot.votes.length)) {
        renderEmpty(node, 'No messages or votes yet.');
        statsNode.textContent = "";
        return;
      }

      statsNode.textContent = snapshot.messages.length + " Msgs";
      const adminRoles = ["skynet", "librarian", "writer", "tron"];
      
      let html = '';
      let currentRound = null;
      
      function renderVotesForRound(r) {
          if(!snapshot.votes) return '';
          const roundVotes = snapshot.votes.filter(v => v.round_number === r && v.vote_kind === 'termination');
          if(roundVotes.length === 0) return '';
          
          let vHtml = '<div class="chat-bubble system" style="border-color:#bda2f7;">';
          vHtml += '<div class="chat-header"><span class="chat-sender" style="color:#bda2f7">GOVERNANCE VOTE</span><span class="chat-meta">Round ' + r + '</span></div>';
          vHtml += '<details id="vote-det-' + r + '"><summary>View ' + roundVotes.length + ' Termination Votes</summary><div class="chat-content" style="font-family: var(--mono); font-size:12px;">';
          
          roundVotes.forEach(v => {
              const symbol = v.decision === 'yes' || v.decision === 'continue' ? '✅' : (v.decision === 'no' || v.decision === 'close' ? '🛑' : '⚠️');
              vHtml += '<strong style="color:#fff">' + esc(v.voter) + '</strong>: ' + symbol + ' ' + esc(v.decision) + '<br>';
              vHtml += '<span style="color:#aaa">Reason: ' + esc(v.reason) + '</span><br><br>';
          });
          
          vHtml += '</div></details></div>';
          return vHtml;
      }

      snapshot.messages.forEach((message) => {
        if (currentRound !== null && message.round_number !== null && message.round_number !== currentRound) {
            html += renderVotesForRound(currentRound);
        }
        if (message.round_number !== null) {
            currentRound = message.round_number;
        }

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
            let previewText = esc(message.content.substring(0, 100)) + "...";
            if (message.msg_type === "summary") previewText = "Summary Content";
            if (message.turn_kind === "librarian_audit") previewText = "Librarian Audit Log";

            contentHtml = '<details id="msg-det-' + message.id + '"><summary>Expand: ' + previewText + '</summary><div class="chat-content">' + linkCitations(message.content) + '</div></details>';
        } else {
            contentHtml = '<div class="chat-content">' + linkCitations(message.content) + '</div>';
        }

        const metaLabels = [];
        if (message.round_number !== null && message.round_number !== undefined) metaLabels.push('R' + message.round_number);
        if (message.turn_kind && message.turn_kind !== "base") metaLabels.push(message.turn_kind);
        if (message.msg_type !== "standard") metaLabels.push(message.msg_type);

        html += '<div class="' + classes + '">' +
          '<div class="chat-header">' +
          '<span class="chat-sender">' + esc(message.sender) + '</span>' +
          '<span class="chat-meta">' + esc(metaLabels.join(' | ')) + '</span>' +
          '</div>' +
          contentHtml +
          '</div>';
      });
      
      if (currentRound !== null) {
          html += renderVotesForRound(currentRound);
      }
      
      if (snapshot.votes) {
          const admissionVotes = snapshot.votes.filter(v => v.vote_kind === 'candidate_admission');
          if (admissionVotes.length > 0 && currentRound === null) {
              let vHtml = '<div class="chat-bubble system" style="border-color:#bda2f7;">';
              vHtml += '<div class="chat-header"><span class="chat-sender" style="color:#bda2f7">ADMISSION VOTES</span><span class="chat-meta">Pre-debate</span></div>';
              vHtml += '<details id="vote-det-admin"><summary>View ' + admissionVotes.length + ' Admission Votes</summary><div class="chat-content" style="font-family: var(--mono); font-size:12px;">';
              
              admissionVotes.forEach(v => {
                  const symbol = v.decision === 'yes' ? '✅' : (v.decision === 'no' ? '🛑' : '⚠️');
                  vHtml += '<strong style="color:#fff">' + esc(v.voter) + '</strong> (' + esc(v.subject) + '): ' + symbol + ' ' + esc(v.decision) + '<br>';
                  vHtml += '<span style="color:#aaa">Reason: ' + esc(v.reason) + '</span><br><br>';
              });
              vHtml += '</div></details></div>';
              html = vHtml + html;
          }
      }

      node.innerHTML = html;
    }

    function renderKB(snapshot) {
      // 1. Facts
      const factsNode = document.getElementById('tab-facts');
      if (!snapshot.facts || !snapshot.facts.length) {
        renderEmpty(factsNode, 'No accepted facts yet.');
      } else {
        factsNode.innerHTML = snapshot.facts.map(f => 
          '<div class="card">' +
          '<div><span class="label">[F' + f.id + ']</span><span class="label" style="color:var(--fact)">' + esc(f.review_status) + '</span></div>' +
          '<div class="fact-content" style="font-weight:bold; margin-bottom:8px;">' + linkCitations(f.summary || f.content) + '</div>' +
          (f.summary ? '<details id="f-det-' + f.id + '"><summary>View Full Text</summary><div class="card-content" style="margin-top:8px; font-size:12px; border-top:1px solid var(--line); padding-top:8px;">' + linkCitations(f.content) + '</div></details>' : '') +
          '</div>'
        ).join('');
      }
      
      // 2. Claims
      const claimsNode = document.getElementById('tab-claims');
      if (!snapshot.claims || !snapshot.claims.length) {
        renderEmpty(claimsNode, 'No claims generated yet.');
      } else {
        claimsNode.innerHTML = snapshot.claims.map(c => 
          '<div class="card">' +
          '<div><span class="label">[C' + c.id + ']</span></div>' +
          '<div class="fact-content" style="font-weight:bold; margin-bottom:8px;">' + linkCitations(c.summary || c.content) + '</div>' +
          (c.summary ? '<details id="c-det-' + c.id + '"><summary>View Full Text</summary><div class="card-content" style="margin-top:8px; font-size:12px; border-top:1px solid var(--line); padding-top:8px;">' + linkCitations(c.content) + '</div></details>' : '') +
          '</div>'
        ).join('');
      }

      // 3. Candidates
      const candsNode = document.getElementById('tab-cands');
      if (!snapshot.fact_candidates || !snapshot.fact_candidates.length) {
        renderEmpty(candsNode, 'No fact candidates yet.');
      } else {
        candsNode.innerHTML = snapshot.fact_candidates.map(c => {
          let extra = '';
          if (c.status !== 'pending' && c.review_note) {
              extra = '<details id="cand-det-' + c.id + '" style="margin-top:8px; border-color:var(--warning)"><summary>Audit Reason</summary><div style="color:var(--warning); font-size:12px; margin-top:4px;">' + esc(c.review_note) + '</div></details>';
          }
          return '<div class="card">' +
          '<div><span class="label">Cand #' + c.id + '</span><span class="label" style="color:var(--pending)">' + esc(c.status) + '</span></div>' +
          '<div class="candidate-content">' + esc(c.candidate_text) + '</div>' +
          extra +
          '</div>';
        }).join('');
      }
      
      // 4. Web Evidence
      const webNode = document.getElementById('tab-web');
      if (!snapshot.web_evidence || !snapshot.web_evidence.length) {
        renderEmpty(webNode, 'No web searches performed.');
      } else {
        webNode.innerHTML = snapshot.web_evidence.map(w => 
          '<div class="card">' +
          '<div><span class="label">[W' + w.id + ']</span><a href="' + esc(w.url) + '" target="_blank" style="color:var(--accent);font-size:11px;">' + esc(w.source_domain) + '</a></div>' +
          '<div class="card-title">' + esc(w.title) + '</div>' +
          '<div class="card-content">' + esc(w.snippet) + '</div>' +
          '</div>'
        ).join('');
      }
    }

    async function refresh() {
      let snapshot;
      
      if (window.GROX_STATIC_DATA) {
          // Static Mode
          let subId = currentSubtopicId;
          if (!subId && window.GROX_STATIC_DATA.subtopics.length > 0) {
              subId = window.GROX_STATIC_DATA.subtopics[window.GROX_STATIC_DATA.subtopics.length - 1].id;
          }
          if (subId && window.GROX_STATIC_DATA.subtopic_data[subId]) {
              snapshot = window.GROX_STATIC_DATA.subtopic_data[subId];
          } else {
              snapshot = window.GROX_STATIC_DATA.subtopic_data["default"] || window.GROX_STATIC_DATA;
          }
      } else {
          // Dynamic Mode
          let url = '/api/dashboard';
          if(currentSubtopicId) url += '?subtopic_id=' + currentSubtopicId;
          const response = await fetch(url, { cache: 'no-store' });
          snapshot = await response.json();
      }
      
      const detailsState = {};
      document.querySelectorAll('details').forEach(el => {
         if(el.id) detailsState[el.id] = el.open;
      });
      
      // Remember scroll position of timeline
      const timelineScroll = document.getElementById('scroll-timeline').scrollTop;

      updateKnowledgeMap(snapshot);
      renderTopic(snapshot);
      renderPlan(snapshot);
      renderMessages(snapshot);
      renderKB(snapshot);
      
      document.querySelectorAll('details').forEach(el => {
         if(el.id && detailsState[el.id]) el.open = true;
      });
      document.getElementById('scroll-timeline').scrollTop = timelineScroll;
      
      if (window.GROX_STATIC_DATA || (snapshot.topic && snapshot.topic.status === 'Closed')) {
          if (window.refreshInterval) clearInterval(window.refreshInterval);
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
