"""Write the deterministic v5 team-battle sample log, tests/data/v5_team_battle_sample.jsonl.

Regenerate (from the repo root) after any engine change that alters the log:
    python3 -m tests.make_v5_fixture
tests/test_teams.py checks that the file is up to date and that replay.py verifies it.

The episode: 2 teams (red: 3 agents, blue: 2) on a 16 x 8 x 6 platform split at x = 8, on a
simulated clock, with continuous walking (docs/walking.md). Buys, places on each side, team /
public / private chat, finished votes, a red agent crossing the line and removing a blue block
(clears only blue's votes), a failed out-of-reach attack, a hit, an attack on cooldown, a kill in
3 hits (inventory lost, vote dropped), a dead agent's ignored action, a give to a walker mid-path
(out of reach at its destination), a snapshot taken mid-walk (`walks`), a walk truncated by a
block placed on a cell ahead of the walker (`stopped`, `path_offset`), and an all_done end.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blockswarm.clock import SimClock  # noqa: E402
from blockswarm.recorder import Recorder  # noqa: E402
from blockswarm.world import Config, World  # noqa: E402

FIXTURE = ROOT / "tests" / "data" / "v5_team_battle_sample.jsonl"

TEAMS = {"red": ["red_1", "red_2", "red_3"], "blue": ["blue_1", "blue_2"]}


def fixture_config() -> Config:
    return Config(width=16, depth=8, height=6, agents=TEAMS["red"] + TEAMS["blue"], teams=TEAMS,
                  shops={"red": [3, 4, 0], "blue": [12, 4, 0]}, snapshot_interval_s=2.0,
                  walking="continuous")


def build_fixture(path: str | Path | None = None) -> World:
    """Play the scripted episode (writing the log to `path` if given) and return the world.
    Asserts every step's outcome, so a rule change shows up here."""
    clock = SimClock()
    w = World(fixture_config(), recorder=Recorder(path), episode_id="fixture-team_battle", clock=clock)

    def at(t: float, agent: str, tool: str, ok: bool = True, **args):
        clock.advance_to(t)
        r = w.act(agent, tool, **args)
        if r.ok != ok:
            raise AssertionError(f"t={t} {agent} {tool} {args}: expected ok={ok}, got {r}")
        return r

    # start rings: red_1 (2,5,0) red_2 (4,5,0) red_3 (2,3,0); blue_1 (11,5,0) blue_2 (13,5,0)
    at(0.5, "red_1", "buy", item="oak_planks", n=4)
    at(0.8, "blue_1", "buy", item="stone", n=4)
    at(1.0, "red_2", "say", text="Red team: let's build the house just east of our shop.")
    at(1.2, "blue_1", "say_all", text="Hello red team, good luck!")
    at(1.4, "red_3", "whisper", to="blue_2", text="psst, blue_2")
    at(1.6, "red_1", "place", x=2, y=6, z=0, type="oak_planks")
    at(1.9, "red_1", "place", x=3, y=6, z=0, type="oak_planks")
    at(2.2, "blue_1", "place", x=11, y=6, z=0, type="stone")
    at(2.4, "blue_1", "place", x=12, y=6, z=0, type="stone")
    at(2.6, "red_1", "declare_done")
    at(2.8, "red_2", "declare_done")
    at(3.0, "blue_1", "declare_done")
    # red_2 crosses the line and removes a blue block: only blue's votes are cleared
    at(3.2, "red_2", "move_to", x=9, y=6, z=0)
    at(4.6, "red_2", "remove", x=11, y=6, z=0)
    at(4.7, "red_2", "remove", x=12, y=6, z=0)
    at(4.8, "red_2", "place", x=9, y=7, z=0, type="stone")  # a red block on blue's side; keeps 1 stone
    # blue_2 goes after red_2; red_1 swings at blue_1 from too far away
    at(5.0, "blue_2", "move_to", x=10, y=7, z=0)
    at(5.2, "red_1", "attack", ok=False, target="blue_1")
    at(6.0, "blue_2", "attack", target="red_2")                # 20 -> 13
    at(6.3, "blue_2", "attack", ok=False, target="red_2")      # attack cooldown (0.6 s)
    at(6.7, "blue_2", "attack", target="red_2")                # 13 -> 6
    at(7.4, "blue_2", "attack", target="red_2")                # 6 -> 0: killed
    clock.advance_to(7.6)
    r = w.act("red_2", "say", text="I'm still here")            # dead: not logged
    if r.ok or r.message != "you are dead":
        raise AssertionError(f"dead agent acted: {r}")
    at(8.0, "red_3", "say", text="red_2 is gone. Keep building.")
    # blue_2 removes red's block on blue's side: red's blocks_placed_by_team (3, all places) now
    # differs from red blocks standing (2) and blocks_by_team_on_other_side drops to 0
    at(8.2, "blue_2", "remove", x=9, y=7, z=0)
    # continuous walking: red_3 walks east along y = 3 (straight, 8 cells; cell i is reached at
    # 9.6 + i / 4.317). red_1 hands it a plank as it passes (at its destination it would be out of
    # reach); the snapshot at 10.1 catches it mid-walk at (4, 3, 0); blue_1 then blocks (8, 3, 0)
    # ahead of it, so it stops at (7, 3, 0) instead of (10, 3, 0).
    at(9.6, "red_3", "move_to", x=10, y=3, z=0)
    clock.advance_to(9.9)
    if w.positions["red_3"] != (3, 3, 0):
        raise AssertionError(f"red_3 should be at (3, 3, 0) at 9.9, is at {w.positions['red_3']}")
    at(9.9, "red_1", "give", to="red_3", type="oak_planks", n=1)
    at(10.1, "blue_1", "place", x=8, y=3, z=0, type="stone")
    stop = w.recorder.records[-1]["diff"]["agents"].get("red_3", {})
    if not stop.get("stopped") or stop["to"] != [7, 3, 0] or stop["path_offset"] != 2:
        raise AssertionError(f"red_3's walk should stop at (7, 3, 0): {stop}")
    at(10.5, "red_3", "declare_done", ok=False)  # still walking (until 10.99)
    at(11.2, "red_3", "declare_done")
    at(11.4, "blue_1", "declare_done")
    at(11.6, "blue_2", "declare_done")  # every living agent has voted: all_done
    if w.end_reason != "all_done":
        raise AssertionError(f"expected all_done, got {w.end_reason}")
    w.recorder.close()
    return w


if __name__ == "__main__":
    build_fixture(FIXTURE)
    print(f"wrote {FIXTURE.relative_to(ROOT)}")
