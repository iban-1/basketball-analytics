"""Probe which stats nba_api can return for one game (a development check, prints only).

Run:  python -m scripts.probe_nba_stats [GAME_ID]
"""
from __future__ import annotations

import sys
import time

import nba_api.stats.endpoints as E

GAME = sys.argv[1] if len(sys.argv) > 1 else "0042100404"   # 2022 Finals, Game 4
PROBES = {
    "traditional": lambda g: E.BoxScoreTraditionalV3(game_id=g, timeout=40),
    "advanced": lambda g: E.BoxScoreAdvancedV3(game_id=g, timeout=40),
    "four_factors": lambda g: E.BoxScoreFourFactorsV3(game_id=g, timeout=40),
    "misc": lambda g: E.BoxScoreMiscV3(game_id=g, timeout=40),
    "scoring": lambda g: E.BoxScoreScoringV3(game_id=g, timeout=40),
    "usage": lambda g: E.BoxScoreUsageV3(game_id=g, timeout=40),
    "hustle": lambda g: E.BoxScoreHustleV2(game_id=g, timeout=40),
    "player_track": lambda g: E.BoxScorePlayerTrackV3(game_id=g, timeout=40),
    "defensive": lambda g: E.BoxScoreDefensiveV2(game_id=g, timeout=40),
    "matchups": lambda g: E.BoxScoreMatchupsV3(game_id=g, timeout=40),
    "summary": lambda g: E.BoxScoreSummaryV3(game_id=g, timeout=40),
    "playbyplay": lambda g: E.PlayByPlayV3(game_id=g, timeout=40),
}

for name, make in PROBES.items():
    try:
        frames = make(GAME).get_data_frames()
        print(f"\n== {name}: OK, {len(frames)} tables")
        for i, df in enumerate(frames):
            print(f"   table {i}: {df.shape[0]} rows x {df.shape[1]} cols -> {list(df.columns)[:40]}")
    except Exception as exc:  # report, never hide
        print(f"\n== {name}: FAILED {type(exc).__name__}: {str(exc)[:150]}")
    time.sleep(1.0)  # be polite to NBA.com
