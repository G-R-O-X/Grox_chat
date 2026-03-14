from grox_chat.graph import ChatState, dispatcher_node, route_from_dispatcher, stage_dispatcher_node, route_from_stage_dispatcher

def test_dispatcher_node():
    state: ChatState = {
        "topic_id": 1,
        "subtopic_id": 1,
        "plan_id": 1,
        "pending_subtopics": [],
        "pending_turns": [
            {"actor": "dreamer", "turn_kind": "base"},
            {"actor": "scientist", "turn_kind": "base"},
            {"actor": "critic", "turn_kind": "base"},
        ],
        "search_retry_count": 0,
        "dog_target": None,
        "cat_target": None,
        "tron_target": None,
        "current_actor": "",
        "current_turn_kind": "",
        "phase": "opening",
        "subtopic_exhausted": False,
        "round_number": 1,
    }

    new_state = dispatcher_node(state)

    assert new_state["current_actor"] == "dreamer"
    assert new_state["current_turn_kind"] == "base"
    assert new_state["pending_turns"] == [
        {"actor": "scientist", "turn_kind": "base"},
        {"actor": "critic", "turn_kind": "base"},
    ]

def test_route_from_dispatcher():
    # If there's an actor, route to them
    assert route_from_dispatcher({"current_actor": "dreamer", "pending_turns": []}) == "dreamer"

    # If no actor and no queued turns, end of round
    assert route_from_dispatcher({"current_actor": "", "pending_turns": []}) == "end_of_round"


def test_stage_dispatcher_node_pops_first_stage():
    stages = [
        {"agents": [{"actor": "dreamer", "turn_kind": "base"}], "parallel": True},
        {"agents": [{"actor": "critic", "turn_kind": "base"}], "parallel": True},
    ]
    state = {"pending_stages": stages}
    result = stage_dispatcher_node(state)

    assert result["current_stage"] == stages[0]
    assert result["pending_stages"] == [stages[1]]


def test_stage_dispatcher_node_returns_none_when_empty():
    state = {"pending_stages": []}
    result = stage_dispatcher_node(state)

    assert result["current_stage"] is None
    assert result["pending_stages"] == []


def test_route_from_stage_dispatcher_parallel():
    state = {"current_stage": {"agents": [], "parallel": True}}
    assert route_from_stage_dispatcher(state) == "parallel_group"


def test_route_from_stage_dispatcher_sequential():
    state = {"current_stage": {"agents": [], "parallel": False}}
    assert route_from_stage_dispatcher(state) == "sequential_group"


def test_route_from_stage_dispatcher_end_of_round():
    state = {"current_stage": None}
    assert route_from_stage_dispatcher(state) == "end_of_round"
