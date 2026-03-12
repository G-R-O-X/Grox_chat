import asyncio
import random
import time


class GlobalThrottle:
    def __init__(self, delay_seconds: float = 1.0, jitter_min_seconds: float = 0.0, jitter_max_seconds: float = 0.0):
        self.delay_seconds = delay_seconds
        self.jitter_min_seconds = jitter_min_seconds
        self.jitter_max_seconds = jitter_max_seconds
        self._lock = asyncio.Lock()
        self._last_call_time = 0.0

    async def acquire(self):
        """Wait until it is safe to make an API call, maintaining global pacing."""
        async with self._lock:
            target_delay = self.delay_seconds
            if self.jitter_max_seconds > 0.0:
                target_delay += random.uniform(self.jitter_min_seconds, self.jitter_max_seconds)
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < target_delay:
                await asyncio.sleep(target_delay - elapsed)
            self._last_call_time = time.time()


async def _wait_with_jitter(base_seconds: float, jitter_min_seconds: float, jitter_max_seconds: float):
    delay = base_seconds
    if jitter_max_seconds > 0.0:
        delay += random.uniform(jitter_min_seconds, jitter_max_seconds)
    if delay > 0.0:
        await asyncio.sleep(delay)

# Provider-specific throttles
_minimax_throttle = GlobalThrottle(
    delay_seconds=0.0,
    jitter_min_seconds=0.0,
    jitter_max_seconds=1.0,
)
_gemini_throttle = GlobalThrottle(
    delay_seconds=0.0,
    jitter_min_seconds=0.0,
    jitter_max_seconds=1.0,
)


async def wait_for_minimax_slot():
    """Wait until it is safe to make a MiniMax API call."""
    await _minimax_throttle.acquire()


async def wait_for_gemini_slot():
    """Wait until it is safe to make a Gemini API call."""
    await _gemini_throttle.acquire()


async def wait_after_minimax_response():
    """Hold MiniMax responses briefly before releasing them downstream."""
    await _wait_with_jitter(0.0, 0.0, 0.0)


async def wait_after_gemini_response():
    """Hold Gemini responses briefly before releasing them downstream."""
    await _wait_with_jitter(0.0, 0.0, 0.0)


async def wait_for_slot():
    """Compatibility alias for MiniMax throttling."""
    await wait_for_minimax_slot()
