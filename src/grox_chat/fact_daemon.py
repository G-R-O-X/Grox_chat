"""Background fact extraction and librarian review daemon."""

import asyncio
import json
import logging
from typing import Optional

from . import api
from .agents import SKYNET, SPECTATOR
from .broker import call_text
from .embedding import aget_embedding
from .librarian_processor import (
    apply_claim_review,
    apply_librarian_review,
    parse_claim_review,
    parse_librarian_review,
)
from .prompts import PROMPTS
from .rag import build_query_rag_context
from .structured_retry import generate_summary, usable_text_output
from .writer_processor import process_clerk_claim_output, process_writer_output

logger = logging.getLogger(__name__)

# Re-use constants from server module -- import lazily to avoid circular deps
_INLINE_FACT_LIMIT = 1
_SOURCED_FACT_LIMIT = 2
_CLAIM_LIMIT = 2


class FactDaemon:
    """Continuously polls for new messages and pending candidates,
    extracting and reviewing facts in the background."""

    def __init__(self, topic_id: int, subtopic_id: int):
        self.topic_id = topic_id
        self.subtopic_id = subtopic_id
        self._shutdown = asyncio.Event()
        self._drain = asyncio.Event()
        self._drain_complete = asyncio.Event()
        self._clerk_done = asyncio.Event()
        self._clerk_task: Optional[asyncio.Task] = None
        self._librarian_task: Optional[asyncio.Task] = None

    async def start(self):
        self._clerk_task = asyncio.create_task(
            self._clerk_loop(),
            name=f"fact-clerk-{self.topic_id}-{self.subtopic_id}",
        )
        self._librarian_task = asyncio.create_task(
            self._librarian_loop(),
            name=f"fact-librarian-{self.topic_id}-{self.subtopic_id}",
        )
        logger.info(
            "[daemon] Started fact daemon for topic=%s subtopic=%s",
            self.topic_id,
            self.subtopic_id,
        )

    async def _clerk_loop(self):
        """Poll for new standard messages, extract FactCandidates + ClaimCandidates."""
        last_processed_id = 0
        while not self._shutdown.is_set():
            try:
                new_msgs = api.get_messages_since(
                    self.topic_id,
                    self.subtopic_id,
                    last_processed_id,
                    "standard",
                )
                for msg in new_msgs:
                    await self._extract_candidates(msg)
                    last_processed_id = max(last_processed_id, msg["id"])
                if self._drain.is_set() and not new_msgs:
                    break
            except Exception as exc:
                logger.warning("[daemon] Clerk loop error: %s", exc)
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=5.0)
                break  # shutdown was set
            except asyncio.TimeoutError:
                pass  # just a sleep substitute
        self._clerk_done.set()

    async def _librarian_loop(self):
        """Poll for pending candidates, review them."""
        while not self._shutdown.is_set():
            try:
                pending = api.get_pending_fact_candidates(
                    self.topic_id, self.subtopic_id
                )
                pending_claims = api.get_pending_claim_candidates(
                    self.topic_id, self.subtopic_id
                )
                if pending or pending_claims:
                    await self._review_batch(pending, pending_claims)
                elif self._drain.is_set() and self._clerk_done.is_set():
                    # Final sweep after clerk is done to catch any late candidates
                    pending = api.get_pending_fact_candidates(
                        self.topic_id, self.subtopic_id
                    )
                    pending_claims = api.get_pending_claim_candidates(
                        self.topic_id, self.subtopic_id
                    )
                    if pending or pending_claims:
                        try:
                            await self._review_batch(pending, pending_claims)
                        except Exception as exc:
                            logger.warning("[daemon] Final drain sweep failed: %s", exc)
                    self._drain_complete.set()
                    break
            except Exception as exc:
                logger.warning("[daemon] Librarian loop error: %s", exc)
                if self._drain.is_set() and self._clerk_done.is_set():
                    logger.warning("[daemon] Review failed during drain; completing drain anyway")
                    self._drain_complete.set()
                    break
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=8.0)
                break
            except asyncio.TimeoutError:
                pass

    async def drain_and_stop(self, timeout: float = 90.0):
        """Drain all pending work, then stop."""
        self._drain.set()
        try:
            await asyncio.wait_for(self._drain_complete.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning("[daemon] Drain timed out after %.0fs", timeout)
        self._shutdown.set()
        # Wait for tasks to finish
        tasks = [t for t in (self._clerk_task, self._librarian_task) if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(
            "[daemon] Stopped fact daemon for topic=%s subtopic=%s",
            self.topic_id,
            self.subtopic_id,
        )

    async def stop(self):
        """Immediately stop without draining."""
        self._shutdown.set()
        tasks = [t for t in (self._clerk_task, self._librarian_task) if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _extract_candidates(self, msg: dict):
        """Extract fact and claim candidates from a single message."""
        sender = msg.get("sender", "")
        content = msg.get("content", "")
        if not content or sender in {SKYNET, "writer", "librarian", "fact_proposer"}:
            return
        if sender == SPECTATOR:
            return

        # Import server functions lazily to avoid circular imports
        from .server import (
            _extract_number_fact_candidates,
            _normalize_clerk_fact_candidates_contract,
            _normalize_clerk_claim_candidates_contract,
            _fact_candidates_output_is_usable,
            _claim_candidates_output_is_usable,
            _extract_fact_ids_from_text,
            _render_fact_lookup_context,
            _call_text_with_structured_retry,
            FACT_CITATION_PROTOCOL,
            NUMBER_FACT_LIMIT,
        )

        # Number facts
        number_candidates = _extract_number_fact_candidates([msg])[
            :NUMBER_FACT_LIMIT
        ]
        if number_candidates:
            await process_writer_output(
                self.topic_id,
                self.subtopic_id,
                None,
                "",
                structured_facts=number_candidates,
                fact_stage="synthesized",
                round_number=msg.get("round_number"),
                max_candidates=NUMBER_FACT_LIMIT,
            )

        # Sourced facts via LLM
        try:
            rag_context, _ = await build_query_rag_context(
                self.topic_id, content[:500]
            )
        except Exception:
            rag_context = ""

        sourced_prompt = (
            f"Topic context:\n{FACT_CITATION_PROTOCOL}\n"
            f"Message from {sender}:\n{content[:2000]}\n"
            f"{rag_context}\n"
            f"\nTASK: Extract at most {_SOURCED_FACT_LIMIT} externally-sourced "
            "conclusion candidates from this message. "
            "Only include claims that look like paper conclusions, official "
            "statistics, reputable web conclusions, or expert-source claims. "
            "Reply with strict JSON only: "
            '{"action":"propose_fact_candidates",'
            '"fact_candidates":[{"candidate_text":"...",'
            '"source_refs_json":["..."],"source_excerpt":"..."}]}.'
        )
        sourced_text = await _call_text_with_structured_retry(
            stage_name=f"Daemon sourced fact pass msg={msg['id']}",
            validator=_fact_candidates_output_is_usable,
            invoke=lambda: call_text(
                sourced_prompt,
                provider="minimax",
                strategy="react",
                allow_web=True,
                system_instruction=PROMPTS["fact_proposer"],
                fallback_role="fact_proposer",
                require_json=True,
                topic_id=self.topic_id,
                subtopic_id=self.subtopic_id,
            ),
        )
        if sourced_text:
            parsed_sourced = _normalize_clerk_fact_candidates_contract(
                sourced_text
            )
            if (
                parsed_sourced["parsed_ok"]
                and parsed_sourced["fact_candidates"]
            ):
                await process_writer_output(
                    self.topic_id,
                    self.subtopic_id,
                    None,
                    "",
                    structured_facts=parsed_sourced["fact_candidates"],
                    fact_stage="synthesized",
                    round_number=msg.get("round_number"),
                    max_candidates=_SOURCED_FACT_LIMIT,
                )

        # Claim candidates from cited facts
        cited_fact_ids = sorted(set(_extract_fact_ids_from_text(content)))
        if cited_fact_ids:
            support_facts = api.get_facts_by_ids(
                self.topic_id, cited_fact_ids
            )
            if support_facts:
                claim_prompt = (
                    f"Topic context:\n{FACT_CITATION_PROTOCOL}\n"
                    f"Message from {sender}:\n{content[:2000]}\n"
                    f"=== VERIFIED FACTS REFERENCED ===\n"
                    f"{_render_fact_lookup_context(support_facts)}\n"
                    f"\nTASK: Extract at most {_CLAIM_LIMIT} derived claim "
                    "candidates that are explicitly supported by cited "
                    "facts [F...]. "
                    "Reply with strict JSON only: "
                    '{"action":"propose_claim_candidates",'
                    '"claim_candidates":[{"candidate_text":"...",'
                    '"support_fact_ids_json":[1,2],'
                    '"rationale_short":"..."}]}.'
                )
                claim_text = await _call_text_with_structured_retry(
                    stage_name=f"Daemon claim pass msg={msg['id']}",
                    validator=_claim_candidates_output_is_usable,
                    invoke=lambda: call_text(
                        claim_prompt,
                        provider="minimax",
                        strategy="direct",
                        allow_web=False,
                        system_instruction=PROMPTS["fact_proposer"],
                        fallback_role="fact_proposer",
                        require_json=True,
                    ),
                )
                if claim_text:
                    parsed_claims = (
                        _normalize_clerk_claim_candidates_contract(claim_text)
                    )
                    if (
                        parsed_claims["parsed_ok"]
                        and parsed_claims["claim_candidates"]
                    ):
                        await process_clerk_claim_output(
                            self.topic_id,
                            self.subtopic_id,
                            None,
                            parsed_claims["claim_candidates"],
                            max_candidates=_CLAIM_LIMIT,
                        )

    async def _review_batch(self, facts: list, claims: list):
        """Review pending fact and claim candidates."""
        from .server import (
            _query_librarian_review_text,
            build_librarian_prompt,
            build_claim_review_prompt,
            _extract_fact_ids_from_text,
        )

        topic = api.get_topic(self.topic_id)
        subtopic = api.get_subtopic(self.subtopic_id)
        if not topic or not subtopic:
            return

        messages = api.get_messages(
            self.topic_id, subtopic_id=self.subtopic_id, limit=12
        )
        recent_message_ids = [m["id"] for m in messages if "id" in m]

        for candidate in facts:
            try:
                rag_context, _ = await build_query_rag_context(
                    self.topic_id,
                    candidate["candidate_text"],
                    exclude_ids=recent_message_ids,
                )
                # Create a minimal state-like dict for the prompt builder
                fake_state = {
                    "round_number": 0,
                    "phase": "debate",
                }
                prompt = build_librarian_prompt(
                    fake_state,
                    topic,
                    subtopic,
                    candidate,
                    messages,
                    rag_context,
                )

                resp_text, provider = await _query_librarian_review_text(
                    prompt,
                    stage_name=f"Daemon fact review {candidate['id']}",
                    validator=lambda text: usable_text_output(text)
                    and bool(_extract_json(text)),
                )
                try:
                    review = parse_librarian_review(
                        resp_text, candidate["candidate_text"]
                    )
                except Exception:
                    if provider != "minimax":
                        raise
                    logger.warning(
                        "[daemon] MiniMax review for candidate %s failed; "
                        "retrying with Gemini.",
                        candidate["id"],
                    )
                    resp_text = await call_text(
                        prompt,
                        provider="gemini",
                        strategy="direct",
                        allow_web=True,
                        system_instruction=PROMPTS["librarian"],
                        model="gemini-3.0-flash",
                        temperature=0.7,
                        max_tokens=8192,
                        fallback_role="librarian",
                    )
                    review = parse_librarian_review(
                        resp_text, candidate["candidate_text"]
                    )
                result = await apply_librarian_review(
                    self.topic_id, candidate, review
                )
                fact_id = result.get("accepted_fact_id")
                stored_text = result.get("stored_text")
                if fact_id and stored_text:
                    try:
                        summary = await generate_summary(stored_text)
                        if summary:
                            emb = await aget_embedding(summary)
                            if emb:
                                api.update_fact_summary_and_embedding(
                                    fact_id, summary, emb
                                )
                    except Exception as exc:
                        logger.warning(
                            "[daemon] Summary generation failed for "
                            "fact %s: %s",
                            fact_id,
                            exc,
                        )
            except Exception as exc:
                logger.warning(
                    "[daemon] Failed to review candidate %s: %s",
                    candidate["id"],
                    exc,
                )

        for candidate in claims:
            try:
                support_ids = []
                try:
                    if isinstance(
                        candidate.get("support_fact_ids_json"), str
                    ):
                        support_ids = [
                            int(item)
                            for item in json.loads(
                                candidate["support_fact_ids_json"] or "[]"
                            )
                        ]
                except Exception:
                    support_ids = (
                        _extract_fact_ids_from_text(
                            candidate.get("support_fact_ids_json", "")
                        )
                        if isinstance(
                            candidate.get("support_fact_ids_json"), str
                        )
                        else []
                    )

                support_facts = api.get_facts_by_ids(
                    self.topic_id, support_ids
                )
                if not support_facts:
                    api.update_claim_candidate_review(
                        candidate["id"],
                        "reject",
                        review_note="No valid support facts were available "
                        "for review.",
                    )
                    continue

                rag_context, _ = await build_query_rag_context(
                    self.topic_id,
                    candidate["candidate_text"],
                    exclude_ids=recent_message_ids,
                )
                fake_state = {"round_number": 0, "phase": "debate"}
                prompt = build_claim_review_prompt(
                    fake_state,
                    topic,
                    subtopic,
                    candidate,
                    messages,
                    support_facts,
                    rag_context,
                )

                resp_text, provider = await _query_librarian_review_text(
                    prompt,
                    stage_name=f"Daemon claim review {candidate['id']}",
                    validator=lambda text: usable_text_output(text)
                    and bool(_extract_json(text)),
                )
                try:
                    review = parse_claim_review(
                        resp_text,
                        candidate["candidate_text"],
                        support_ids,
                    )
                except Exception:
                    if provider != "minimax":
                        raise
                    resp_text = await call_text(
                        prompt,
                        provider="gemini",
                        strategy="direct",
                        allow_web=True,
                        system_instruction=PROMPTS["librarian"],
                        model="gemini-3.0-flash",
                        temperature=0.7,
                        max_tokens=8192,
                        fallback_role="librarian",
                        require_json=True,
                    )
                    review = parse_claim_review(
                        resp_text,
                        candidate["candidate_text"],
                        support_ids,
                    )
                result = await apply_claim_review(
                    self.topic_id, candidate, review
                )
                claim_id = result.get("accepted_claim_id")
                stored_text = result.get("stored_text")
                if claim_id and stored_text:
                    try:
                        summary = await generate_summary(
                            stored_text, max_words=30
                        )
                        if summary:
                            api.update_claim_summary(claim_id, summary)
                    except Exception as exc:
                        logger.warning(
                            "[daemon] Summary generation failed for "
                            "claim %s: %s",
                            claim_id,
                            exc,
                        )
            except Exception as exc:
                logger.warning(
                    "[daemon] Failed to review claim candidate %s: %s",
                    candidate["id"],
                    exc,
                )


def _extract_json(text: str):
    """Lightweight JSON extraction for validation."""
    from .json_utils import extract_json_object

    return extract_json_object(text)


# ---------------------------------------------------------------------------
# Module-level daemon registry
# ---------------------------------------------------------------------------

_active_daemons: dict[tuple[int, int], FactDaemon] = {}


def get_active_daemon(
    topic_id: int, subtopic_id: int
) -> Optional[FactDaemon]:
    """Return the running daemon for a given topic/subtopic pair, if any."""
    return _active_daemons.get((topic_id, subtopic_id))


async def start_daemon(topic_id: int, subtopic_id: int) -> FactDaemon:
    """Create and start a new FactDaemon, stopping any existing one first."""
    key = (topic_id, subtopic_id)
    if key in _active_daemons:
        await _active_daemons[key].stop()
    daemon = FactDaemon(topic_id, subtopic_id)
    _active_daemons[key] = daemon
    await daemon.start()
    return daemon


async def stop_daemon(topic_id: int, subtopic_id: int):
    """Immediately stop a running daemon."""
    key = (topic_id, subtopic_id)
    daemon = _active_daemons.pop(key, None)
    if daemon:
        await daemon.stop()


async def drain_daemon(
    topic_id: int, subtopic_id: int, timeout: float = 90.0
):
    """Drain pending work and then stop the daemon."""
    key = (topic_id, subtopic_id)
    daemon = _active_daemons.get(key)
    if daemon:
        try:
            await daemon.drain_and_stop(timeout)
        finally:
            _active_daemons.pop(key, None)
