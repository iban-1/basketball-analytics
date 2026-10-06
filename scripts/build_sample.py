"""Build the bundled demo sample: every NBA Finals game of 2016-2022 (about 40 games, ~4 MB).

Run:  python -m scripts.build_sample
Fetches each game from NBA.com (politely, through the normal cache), then copies its tables into
data/sample/<game_id>/ and writes data/sample/games.csv. The website falls back to this sample when NBA.com
does not answer, which is what happens on public cloud hosts. The data belongs to NBA.com; see the README.
"""
from __future__ import annotations

import shutil

import pandas as pd

from src.bball.config import ROOT, load_config
from src.bball.nba_stats import _cache_dir, list_games, load_game_tables

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2022)]
OUT = ROOT / "data" / "sample"


def is_finals(game_id: str) -> bool:
    """Playoff ids are 004YY00RSG: R = round (4 = Finals), S = series, G = game."""
    return len(game_id) == 10 and game_id[7] == "4"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for season in SEASONS:
        g = list_games(season, "Playoffs")
        rows.append(g[g["game_id"].map(is_finals)].assign(season=season))
    games = pd.concat(rows, ignore_index=True)
    print(f"{len(games)} Finals games:", games.groupby("season").size().to_dict(), flush=True)
    done = 0
    for r in games.itertuples():
        load_game_tables(r.game_id)                       # fetch (or read the local cache)
        dst = OUT / r.game_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(_cache_dir(r.game_id), dst)
        done += 1
        print(f"[{done}/{len(games)}] {r.label}", flush=True)
    games.to_csv(OUT / "games.csv", index=False)
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT}  ({size / 1e6:.1f} MB, {len(games)} games)")


if __name__ == "__main__":
    load_config()
    main()
