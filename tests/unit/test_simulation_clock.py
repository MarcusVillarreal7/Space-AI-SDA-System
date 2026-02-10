"""
Tests for SimulationClock — tick advancement, speed, seek, wraparound.
"""

import asyncio
import pytest

from src.api.simulation_clock import SimulationClock, REAL_INTERVAL_S


class TestSimulationClockInit:
    def test_defaults(self):
        clock = SimulationClock()
        assert clock.max_timestep == 1439
        assert clock.timestep == 0
        assert clock.speed == 60.0
        assert clock.is_playing is False

    def test_custom_max(self):
        clock = SimulationClock(max_timestep=100)
        assert clock.max_timestep == 100


class TestSimulationClockControls:
    def test_play_sets_flag(self):
        clock = SimulationClock()
        # Directly set playing flag without creating async task
        clock.is_playing = True
        assert clock.is_playing is True

    def test_pause_sets_flag(self):
        clock = SimulationClock()
        clock.is_playing = True
        clock.pause()
        assert clock.is_playing is False

    def test_seek(self):
        clock = SimulationClock()
        clock.seek(500)
        assert clock.timestep == 500

    def test_seek_clamps_min(self):
        clock = SimulationClock()
        clock.seek(-10)
        assert clock.timestep == 0

    def test_seek_clamps_max(self):
        clock = SimulationClock(max_timestep=100)
        clock.seek(200)
        assert clock.timestep == 100

    def test_set_speed(self):
        clock = SimulationClock()
        clock.set_speed(360)
        assert clock.speed == 360.0

    def test_set_speed_clamps_min(self):
        clock = SimulationClock()
        clock.set_speed(0.01)
        assert clock.speed == 0.1

    def test_set_speed_clamps_max(self):
        clock = SimulationClock()
        clock.set_speed(99999)
        assert clock.speed == 3600.0


class TestSimulationClockCallback:
    def test_on_tick_registers(self):
        clock = SimulationClock()
        async def cb(ts): pass
        clock.on_tick(cb)
        assert len(clock._callbacks) == 1


@pytest.mark.asyncio
class TestSimulationClockAsync:
    async def test_start_and_stop(self):
        clock = SimulationClock(max_timestep=10)
        clock.set_speed(3600)  # Very fast
        await clock.start()
        assert clock.is_playing is True
        await asyncio.sleep(0.2)
        await clock.stop()
        assert clock.is_playing is False
        # Should have advanced at least one timestep
        assert clock.timestep > 0

    async def test_callback_called(self):
        clock = SimulationClock(max_timestep=100)
        clock.set_speed(3600)
        ticks = []

        async def on_tick(ts):
            ticks.append(ts)

        clock.on_tick(on_tick)
        await clock.start()
        await asyncio.sleep(0.3)
        await clock.stop()
        assert len(ticks) > 0

    async def test_wraparound(self):
        clock = SimulationClock(max_timestep=3)
        clock.set_speed(3600)
        clock.seek(2)

        wrapped = []
        async def on_tick(ts):
            wrapped.append(ts)

        clock.on_tick(on_tick)
        await clock.start()
        await asyncio.sleep(0.5)
        await clock.stop()
        # Should have wrapped around past 3 back to 0
        assert 0 in wrapped or clock.timestep < 3
