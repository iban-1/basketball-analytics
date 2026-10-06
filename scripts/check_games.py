"""Robustness check: build every stat tab for random playoff games from 2016 to 2022.

Run:  python -m scripts.check_games [N]      (default 14 games, seeded, ~1 s per request)
Reports, per game, which tabs came out empty. A failure to fetch or build is shown, not hidden.
"""
from __future__ import annotations

import random
import sys

from src.bball.config import load_config, set_seed
from src.bball.nba_stats import list_games, load_game_tables
from src.bball.tables import build_tabs, did_not_play, game_info, team_summary

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2022)]


def main(n: int = 14) -> None:
    seed = load_config()["seed"]
    set_seed(seed)
    rng = random.Random(seed)
    all_games = [g for s in SEASONS for g in list_games(s, "Playoffs").to_dict("records")]
    print(f"{len(all_games)} playoff games available in {len(SEASONS)} seasons")
    problems = 0
    for g in rng.sample(all_games, n):
        try:
            tabs = build_tabs(tables := load_game_tables(g["game_id"]))
            empty = [name for name, d in tabs.items() if not d]
            summ = team_summary(tables)
            info = game_info(tables)
            dnp = did_not_play(tables)
            flags = f"empty tabs: {empty or 'none'} | summary {'ok' if summ is not None else 'MISSING'} " \
                    f"| arena {'ok' if 'arena' in info else 'MISSING'} | DNP rows {len(dnp)}"
            problems += bool(empty) or summ is None
        except Exception as exc:
            flags = f"FAILED {type(exc).__name__}: {str(exc)[:100]}"
            problems += 1
        print(f"{g['game_id']}  {g['label']:<42s}  {flags}")
    print(f"\n{problems} of {n} sampled games had a problem")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
