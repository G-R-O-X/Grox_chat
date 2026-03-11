from typing import Any, Dict, List, Optional
from typing_extensions import NotRequired, TypedDict

class TurnSpec(TypedDict):
    actor: str
    turn_kind: str


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
    pending_fact_reviews_remaining: NotRequired[bool]
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
