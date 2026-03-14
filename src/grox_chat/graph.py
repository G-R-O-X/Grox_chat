from typing import Any, Dict, List, Optional
from typing_extensions import NotRequired, TypedDict

class TurnSpec(TypedDict):
    actor: str
    turn_kind: str


class StageSpec(TypedDict):
    agents: List[TurnSpec]
    parallel: bool


class ChatState(TypedDict):
    topic_id: int
    plan_id: int
    subtopic_id: int
    pending_subtopics: List[Dict[str, Any]]

    # Execution Round State
    pending_turns: List[TurnSpec]
    current_actor: str
    current_turn_kind: str
    search_retry_count: int

    # Stage-based execution
    pending_stages: NotRequired[List[Dict[str, Any]]]
    current_stage: NotRequired[Optional[Dict[str, Any]]]

    # DAPA Mechanics
    dog_target: Optional[str]
    cat_target: Optional[str]
    tron_target: Optional[str]
    spectator_target: Optional[str]
    spectator_web_boost_target: Optional[str]

    # Internal routing markers
    phase: str
    subtopic_exhausted: bool
    round_number: int
    latest_summary_msg_id: NotRequired[Optional[int]]
    last_writer_round: NotRequired[Optional[int]]
    last_fact_proposer_round: NotRequired[Optional[int]]
    last_final_fact_proposer_round: NotRequired[Optional[int]]
    last_summary_round: NotRequired[Optional[int]]
    pending_fact_reviews_remaining: NotRequired[bool]
    final_librarian_reentry_count: NotRequired[int]
    subtopic_vote_cycle: NotRequired[int]

def dispatcher_node(state: ChatState) -> dict:
    """
    Pops the next turn from the pending_turns queue and sets current_actor/current_turn_kind.
    """
    pending = list(state.get("pending_turns", []))
    actor = ""
    turn_kind = ""
    if pending:
        turn = pending.pop(0)
        actor = turn.get("actor", "")
        turn_kind = turn.get("turn_kind", "base")

    return {
        "current_actor": actor,
        "current_turn_kind": turn_kind,
        "pending_turns": pending,
    }

def route_from_dispatcher(state: ChatState) -> str:
    """
    Routes to the next agent node, or to the end of round logic if queue is empty.
    """
    actor = state.get("current_actor", "")
    if actor:
        return actor
    return "end_of_round"


def stage_dispatcher_node(state: ChatState) -> dict:
    """Pops the next stage from pending_stages."""
    stages = list(state.get("pending_stages", []))
    if not stages:
        return {"current_stage": None, "pending_stages": []}
    stage = stages.pop(0)
    return {"current_stage": stage, "pending_stages": stages}


def route_from_stage_dispatcher(state: ChatState) -> str:
    """Routes to parallel_group, sequential_group, or end_of_round."""
    stage = state.get("current_stage")
    if stage is None:
        return "end_of_round"
    if stage.get("parallel"):
        return "parallel_group"
    return "sequential_group"
