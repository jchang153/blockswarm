"""Threaded live runner: each agent runs in its own thread against the world's (wall) clock."""

from __future__ import annotations

import threading
import time
from typing import Callable, Union

from blockswarm.world import World

# A policy is either an object with `step(world, agent) -> bool` or such a callable.
# `step` makes zero or more tool calls and returns False when the agent wants to stop.
StepFn = Callable[[World, str], bool]
Policy = Union[StepFn, object]


def _step_fn(policy: Policy) -> StepFn:
    step = getattr(policy, "step", None)
    return step if callable(step) else policy  # type: ignore[return-value]


def run_live(world: World, policies: Policy | dict[str, Policy], think_s: float = 0.0,
             poll_s: float = 0.05) -> World:
    """Run every agent concurrently until the episode ends.

    `policies` is one policy shared by all agents or a dict agent -> policy. `think_s` is a
    sleep between an agent's steps. The main thread polls `check_end` (for the time limit).
    If every agent thread stops before the episode ends, or a policy raises, the episode is
    finished as "aborted" (the first exception is re-raised).
    """
    if not isinstance(policies, dict):
        policies = {a: policies for a in world.agents}
    errors: list[BaseException] = []

    def worker(agent: str, step: StepFn) -> None:
        try:
            while not world.done:
                if not step(world, agent):
                    return
                if think_s:
                    time.sleep(think_s)
        except BaseException as e:  # noqa: BLE001 - reported to the caller below
            errors.append(e)
            world.finish("aborted")

    threads = [threading.Thread(target=worker, args=(a, _step_fn(policies[a])), name=f"agent-{a}", daemon=True)
               for a in world.agents]
    for th in threads:
        th.start()
    while any(th.is_alive() for th in threads) and not world.check_end():
        time.sleep(poll_s)
    for th in threads:
        th.join()
    if not world.check_end():
        world.finish("aborted")
    if errors:
        raise errors[0]
    return world
