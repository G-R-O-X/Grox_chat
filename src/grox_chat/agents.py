from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .llm_router import query_with_fallback
from .minimax_client import query_minimax
from .prompts import PROMPTS
from .tools import react_search_loop

ORCHESTRATOR = "orchestrator"
DELIBERATOR = "deliberator"
SPECIAL = "special"
NPC = "npc"

SKYNET = "skynet"
SPECTATOR = "spectator"

DELIBERATORS = [
    "dreamer",
    "scientist",
    "engineer",
    "analyst",
    "critic",
    "contrarian",
]

SPECIALS = ["cat", "dog", "tron", SPECTATOR]
NPCS = ["writer", "fact_proposer", "librarian"]
VOTING_PARTICIPANTS = [SKYNET] + DELIBERATORS + SPECIALS
TARGETABLE_DELIBERATORS = set(DELIBERATORS)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    agent_class: str
    role_prompt: str
    default_provider: str = "minimax"
    default_strategy: str = "direct"
    default_tools: tuple[str, ...] = ()
    can_vote: bool = False
    can_target: bool = False
    can_be_targeted: bool = False


def _build_registry() -> dict[str, AgentSpec]:
    specs = {
        SKYNET: AgentSpec(
            name=SKYNET,
            agent_class=ORCHESTRATOR,
            role_prompt=PROMPTS["skynet"],
            default_provider="gemini",
            default_strategy="direct",
            can_vote=True,
        ),
        "dreamer": AgentSpec(
            name="dreamer",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["dreamer"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "scientist": AgentSpec(
            name="scientist",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["scientist"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "engineer": AgentSpec(
            name="engineer",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["engineer"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "analyst": AgentSpec(
            name="analyst",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["analyst"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "critic": AgentSpec(
            name="critic",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["critic"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "contrarian": AgentSpec(
            name="contrarian",
            agent_class=DELIBERATOR,
            role_prompt=PROMPTS["contrarian"],
            can_vote=True,
            can_be_targeted=True,
        ),
        "cat": AgentSpec(
            name="cat",
            agent_class=SPECIAL,
            role_prompt=PROMPTS["cat"],
            can_vote=True,
            can_target=True,
        ),
        "dog": AgentSpec(
            name="dog",
            agent_class=SPECIAL,
            role_prompt=PROMPTS["dog"],
            can_vote=True,
            can_target=True,
        ),
        "tron": AgentSpec(
            name="tron",
            agent_class=SPECIAL,
            role_prompt=PROMPTS["tron"],
            can_vote=True,
            can_target=True,
        ),
        SPECTATOR: AgentSpec(
            name=SPECTATOR,
            agent_class=SPECIAL,
            role_prompt=PROMPTS["spectator"],
            can_vote=True,
            can_target=True,
        ),
        "writer": AgentSpec(
            name="writer",
            agent_class=NPC,
            role_prompt=PROMPTS["writer"],
            default_provider="gemini",
        ),
        "fact_proposer": AgentSpec(
            name="fact_proposer",
            agent_class=NPC,
            role_prompt=PROMPTS["fact_proposer"],
        ),
        "librarian": AgentSpec(
            name="librarian",
            agent_class=NPC,
            role_prompt=PROMPTS["librarian"],
        ),
    }
    return specs


AGENT_REGISTRY = _build_registry()


def get_agent_spec(name: str) -> AgentSpec:
    return AGENT_REGISTRY[name]


def can_special_target(target: str) -> bool:
    return target in TARGETABLE_DELIBERATORS


def voting_agents() -> list[str]:
    return [name for name in VOTING_PARTICIPANTS if AGENT_REGISTRY[name].can_vote]


def ordinary_deliberators() -> list[str]:
    return list(DELIBERATORS)


def special_agents() -> list[str]:
    return list(SPECIALS)


def is_deliberator(name: str) -> bool:
    return AGENT_REGISTRY.get(name, AgentSpec("", "", "")).agent_class == DELIBERATOR


def is_special(name: str) -> bool:
    return AGENT_REGISTRY.get(name, AgentSpec("", "", "")).agent_class == SPECIAL


def is_npc(name: str) -> bool:
    return AGENT_REGISTRY.get(name, AgentSpec("", "", "")).agent_class == NPC


def _extract_json_vote(text: str) -> Optional[dict]:
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    import re

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_vote_response(text: str) -> bool:
    parsed = _extract_json_vote(text)
    if isinstance(parsed, dict):
        raw_vote = parsed.get("vote")
        if isinstance(raw_vote, bool):
            return raw_vote
        if isinstance(raw_vote, str):
            normalized = raw_vote.strip().lower()
            if normalized in {"yes", "true", "continue", "approve", "select", "replan", "close"}:
                return True
            if normalized in {"no", "false", "reject", "skip", "deny"}:
                return False
    normalized_text = (text or "").strip().lower()
    return normalized_text.startswith("yes") or normalized_text.startswith("true")


class Agent:
    def __init__(self, spec: AgentSpec):
        self.spec = spec

    async def call(
        self,
        prompt: str,
        *,
        allow_web: bool = False,
        require_json: bool = False,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        boost: Optional[str] = None,
    ) -> str:
        del require_json
        role_prompt = self.spec.role_prompt
        if boost:
            role_prompt = f"{role_prompt}\n\nBOOST:\n{boost}"

        if self.spec.default_provider == "gemini":
            return await query_with_fallback(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_instruction=role_prompt,
                use_google_search=allow_web,
                enable_fallback=True,
                fallback_role=self.spec.name,
            )

        if allow_web:
            final_text, _ = await react_search_loop(
                self.spec.name,
                prompt,
                max_iter=2,
                system_prompt=role_prompt,
            )
            return final_text

        final_text, _ = await query_minimax(
            system_prompt=role_prompt,
            question=prompt,
            model="MiniMax-M2.5",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return final_text

    async def vote(self, prompt: str, *, allow_web: bool = False) -> bool:
        vote_instruction = (
            f"{self.spec.role_prompt}\n\n"
            "VOTING MODE:\n"
            "You are not posting a normal debate message. "
            'You must reply with strict JSON only: {"vote":"yes"} or {"vote":"no"}.'
        )
        if self.spec.default_provider == "gemini":
            text = await query_with_fallback(
                prompt,
                system_instruction=vote_instruction,
                use_google_search=allow_web,
                enable_fallback=True,
                fallback_role=self.spec.name,
            )
        elif allow_web:
            text, _ = await react_search_loop(
                self.spec.name,
                prompt,
                max_iter=2,
                system_prompt=vote_instruction,
            )
        else:
            text, _ = await query_minimax(
                system_prompt=vote_instruction,
                question=prompt,
                model="MiniMax-M2.5",
                temperature=0.7,
                max_tokens=8192,
            )
        return parse_vote_response(text)


def get_agent(name: str) -> Agent:
    return Agent(get_agent_spec(name))
