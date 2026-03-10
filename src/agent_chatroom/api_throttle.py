import asyncio
import time

class GlobalThrottle:
    def __init__(self, delay_seconds: float = 1.0):
        self.delay_seconds = delay_seconds
        self._lock = asyncio.Lock()
        self._last_call_time = 0.0

    async def acquire(self):
        """Wait until it is safe to make an API call, maintaining global pacing."""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self.delay_seconds:
                await asyncio.sleep(self.delay_seconds - elapsed)
            self._last_call_time = time.time()

# Singleton instance
_throttle = GlobalThrottle(delay_seconds=1.5)

async def wait_for_slot():
    """Wait until it is safe to make an API call."""
    await _throttle.acquire()
