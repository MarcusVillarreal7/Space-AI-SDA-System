"""
SimulationClock — Async background timer for simulation playback.

Manages a timestep index (0 to max_timestep) that advances at a
configurable speed multiplier. Supports play/pause/seek and notifies
registered callbacks on each tick.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Coroutine, Any

logger = logging.getLogger(__name__)

# Real-world time between simulation timesteps (1 minute in the dataset)
REAL_INTERVAL_S = 60.0


class SimulationClock:
    """
    Async simulation clock that drives WebSocket position broadcasts.

    The dataset has 1440 timesteps spanning 24 hours (1 per minute).
    At speed=1, the clock advances 1 timestep per 60 seconds (real-time).
    At speed=60, it advances 1 timestep per second.
    At speed=360, it advances 6 timesteps per second.
    """

    def __init__(self, max_timestep: int = 1439):
        self.max_timestep = max_timestep
        self.timestep: int = 0
        self.speed: float = 60.0  # Default: 1 simulated minute per real second
        self.is_playing: bool = False
        self._task: asyncio.Task | None = None
        self._callbacks: list[Callable[[int], Coroutine[Any, Any, None]]] = []

    def on_tick(self, callback: Callable[[int], Coroutine[Any, Any, None]]) -> None:
        """Register an async callback to be called on each tick."""
        self._callbacks.append(callback)

    def play(self) -> None:
        """Start or resume playback."""
        if self.is_playing:
            return
        self.is_playing = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
        logger.info("Clock playing at speed %.1fx, timestep %d", self.speed, self.timestep)

    def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False
        logger.info("Clock paused at timestep %d", self.timestep)

    def seek(self, timestep: int) -> None:
        """Jump to a specific timestep."""
        self.timestep = max(0, min(timestep, self.max_timestep))
        logger.info("Clock seeked to timestep %d", self.timestep)

    def set_speed(self, speed: float) -> None:
        """Set the simulation speed multiplier."""
        self.speed = max(0.1, min(3600.0, speed))
        logger.info("Clock speed set to %.1fx", self.speed)

    async def start(self) -> None:
        """Start the clock background task (called during app lifespan)."""
        self.is_playing = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the clock background task."""
        self.is_playing = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        """Main loop: advance timestep and notify callbacks."""
        while True:
            if not self.is_playing:
                await asyncio.sleep(0.1)
                continue

            # Tick interval: how often to advance one timestep
            # At speed=60, interval = 60/60 = 1 second
            interval = REAL_INTERVAL_S / self.speed

            # Clamp minimum interval to avoid overwhelming the system
            interval = max(interval, 1.0 / 30.0)  # Max 30 fps

            await asyncio.sleep(interval)

            if not self.is_playing:
                continue

            # Advance timestep
            self.timestep += 1
            if self.timestep > self.max_timestep:
                self.timestep = 0  # Wrap around

            # Notify callbacks
            for cb in self._callbacks:
                try:
                    await cb(self.timestep)
                except Exception:
                    logger.exception("Error in clock callback")
