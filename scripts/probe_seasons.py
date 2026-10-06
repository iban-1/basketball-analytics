"""Development check: which stat tables exist for playoff games from 2015-16 to 2021-22?

Run:  python -m scripts.probe_seasons
Prints, per season, the number of playoff games and for one sample game which endpoints
returned non-empty player tables. Nothing is saved except the NBA.com response cache.
"""
from __future__ import annotations

from src.bball.nba_stats import ENDPOINTS, get_endpoint, list_games

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2022)]

for season in SEASONS:
    try:
        games = list_games(season, "Playoffs")
    except Exception as exc:
        print(f"{season}: could not list games: {exc}")
        continue
    sample = games.iloc[-1]          # the last game of the playoffs (a Finals game)
    print(f"\n{season}: {len(games)} playoff games; sample {sample['game_id']} {sample['label']}")
    for name in ENDPOINTS:
        try:
            frames = get_endpoint(sample["game_id"], name)
            sizes = [len(f) for f in frames]
            ok = any(s > 0 for s in sizes)
            print(f"   {name:13s} {'OK   ' if ok else 'EMPTY'} table rows {sizes}")
        except Exception as exc:
            print(f"   {name:13s} FAILED {str(exc)[:110]}")
