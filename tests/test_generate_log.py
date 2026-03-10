import importlib.util
import os
from pathlib import Path

import pytest

from grox_chat import api
from grox_chat.db import get_db_path, init_db


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATE_LOG_PATH = REPO_ROOT / "scripts" / "generate_log.py"


def _load_generate_log_module():
    spec = importlib.util.spec_from_file_location("generate_log", GENERATE_LOG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_get_data_returns_latest_plan_row():
    generate_log = _load_generate_log_module()

    topic_id = api.create_topic("Topic", "Detail")
    api.create_plan(topic_id, '[{"summary":"Old plan","detail":"Old detail"}]', current_index=1)
    api.create_plan(topic_id, '[{"summary":"New plan","detail":"New detail"}]', current_index=0)

    _, plan, _, _ = generate_log.get_data(Path(get_db_path()), topic_id)

    assert "New plan" in plan["content"]


def test_render_log_includes_topic_level_messages():
    generate_log = _load_generate_log_module()

    topic_id = api.create_topic("Topic", "Detail")
    api.create_plan(topic_id, '[{"summary":"Plan item","detail":"Plan detail"}]', current_index=0)
    subtopic_id = api.create_subtopic(topic_id, "Subtopic", "Subtopic detail")
    api.post_message(topic_id, subtopic_id, "dreamer", "Subtopic message")
    api.post_message(topic_id, None, "audience", "Final topic summary", msg_type="summary")

    topic, plan, subtopics, messages = generate_log.get_data(Path(get_db_path()), topic_id)
    rendered = generate_log.render_log(topic, plan, subtopics, messages)

    assert "=== Topic-level Messages ===" in rendered
    assert "Final topic summary" in rendered
    assert "Subtopic message" in rendered
