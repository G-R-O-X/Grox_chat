from agent_chatroom.graph import ChatState, dispatcher_node, route_from_dispatcher

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
