"""When NBA.com does not answer, a game load must fail after ONE short try, not after minutes of retries."""
import pytest

from src.bball import nba_stats


def test_unreachable_nba_fails_after_one_probe(monkeypatch, tmp_path):
    calls = []

    def fake_fetch(name, game_id, retries=3, timeout=None):
        calls.append((name, retries, timeout))
        raise RuntimeError("simulated: NBA.com did not answer")

    monkeypatch.setattr(nba_stats, "_fetch", fake_fetch)
    monkeypatch.setattr(nba_stats, "_read_cached", lambda game_id, name: None)
    with pytest.raises(RuntimeError, match="did not answer"):
        nba_stats.load_game_tables("0099999999")
    assert len(calls) == 1                       # not seven endpoints
    name, retries, timeout = calls[0]
    assert name == "traditional" and retries == 1 and timeout is not None and timeout <= 30
