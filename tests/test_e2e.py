from grox_chat.graph import dispatcher_node, route_from_dispatcher


def test_dispatcher_routes_to_next_agent_then_end_of_round():
    state = {
        "pending_turns": [
            {"actor": "dreamer", "turn_kind": "base"},
            {"actor": "scientist", "turn_kind": "base"},
        ],
        "current_actor": "",
        "current_turn_kind": "",
    }

    first_step = dispatcher_node(state)
    assert first_step["current_actor"] == "dreamer"
    assert first_step["current_turn_kind"] == "base"
    assert first_step["pending_turns"] == [{"actor": "scientist", "turn_kind": "base"}]
    assert route_from_dispatcher(first_step) == "dreamer"

    second_step = dispatcher_node(first_step)
    assert second_step["current_actor"] == "scientist"
    assert second_step["current_turn_kind"] == "base"
    assert second_step["pending_turns"] == []
    assert route_from_dispatcher(second_step) == "scientist"

    end_step = dispatcher_node(second_step)
    assert end_step["current_actor"] == ""
    assert end_step["current_turn_kind"] == ""
    assert end_step["pending_turns"] == []
    assert route_from_dispatcher(end_step) == "end_of_round"
