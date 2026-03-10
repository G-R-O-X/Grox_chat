import os

import pytest

from grox_chat import api
from grox_chat.db import get_db_path, init_db
from grox_chat.web import build_dashboard_snapshot, create_app, render_dashboard_html


@pytest.fixture(autouse=True)
def setup_teardown():
    os.environ["TESTING"] = "1"
    db_path = get_db_path()
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db()
    yield
    if os.path.exists(db_path):
        os.remove(db_path)


def test_dashboard_snapshot_handles_empty_database():
    snapshot = build_dashboard_snapshot()

    assert snapshot["topic"] is None
    assert snapshot["plan"] is None
    assert snapshot["subtopics"] == []
    assert snapshot["messages"] == []
    assert snapshot["facts"] == []
    assert snapshot["fact_candidates"] == []
    assert snapshot["status"]["db_path"].endswith("test_chatroom.db")


def test_dashboard_snapshot_includes_plan_messages_facts_and_pending_candidates():
    topic_id = api.create_topic("Topic", "Detail")
    plan_id = api.create_plan(
        topic_id,
        '[{"summary":"Subtopic A","detail":"Detail A"},{"summary":"Subtopic B","detail":"Detail B"}]',
        current_index=1,
    )
    assert plan_id is not None

    subtopic_id = api.create_subtopic(topic_id, "Subtopic A", "Detail A")
    api.post_message(topic_id, subtopic_id, "audience", "Grounding brief", round_number=1, turn_kind="base")
    api.post_message(topic_id, subtopic_id, "dog", "Please narrow the claim", round_number=2, turn_kind="base")
    api.insert_fact(topic_id, "Accepted fact", "Librarian")
    api.create_fact_candidate(topic_id, subtopic_id, None, "Pending fact")

    snapshot = build_dashboard_snapshot()

    assert snapshot["topic"]["id"] == topic_id
    assert snapshot["plan"]["current_index"] == 1
    assert len(snapshot["plan"]["items"]) == 2
    assert snapshot["current_subtopic"]["id"] == subtopic_id
    assert [message["sender"] for message in snapshot["messages"]] == ["audience", "dog"]
    assert snapshot["status"]["current_round"] == 2
    assert snapshot["status"]["current_phase"] == "evidence"
    assert snapshot["facts"][0]["content"] == "Accepted fact"
    assert snapshot["fact_candidates"][0]["candidate_text"] == "Pending fact"


def test_create_app_registers_read_only_routes():
    app = create_app()
    routes = {(route.method, route.resource.canonical) for route in app.router.routes()}

    assert ("GET", "/") in routes
    assert ("GET", "/api/dashboard") in routes
    assert ("GET", "/api/health") in routes


def test_dashboard_html_escapes_dynamic_content_in_client_renderer():
    html = render_dashboard_html()

    assert "function esc(value)" in html
    assert "esc(message.content)" in html
    assert "esc(fact.content)" in html
    assert "esc(candidate.candidate_text)" in html
