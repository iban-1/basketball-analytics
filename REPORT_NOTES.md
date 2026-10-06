# Report notes: what I built, why, and how to defend it

Numbers are deliberately not repeated here; they live in the generated results block in `README.md`
(written by `scripts/build_summary.py` from the pipeline's own output files) and in `results/video/`.

## 1. The project in one paragraph

A Streamlit site with two halves. The **stats half** takes any NBA playoff game from 2016 to 2022,
fetches NBA.com's official tables (`src/bball/nba_stats.py`, via the free `nba_api` package), and shows
the box score, shooting, advanced, miscellaneous, hustle and tracking tables plus team analysis
(four factors, scoring sources, game flow, shot profile), player analysis and tracking charts
(`src/bball/tables.py`, `src/bball/analysis.py`). The **video half** runs computer vision on one
broadcast clip: YOLO detects and tracks people, the camera view is matched to the painted court so pixels
become court metres, a ball detector trained on the clip itself finds the ball, and rules turn all of that
into passes, interceptions, possession %, and each player's distance and speed. A top-down radar and
counters are drawn onto the video; the dashboard also replays the tracked positions on a court.

## 2. Main design decisions (and the reasoning)

**Official stats for the whole game, vision only for the clip.** NBA.com already publishes accurate
per-game numbers, so the video is not asked to reproduce them. It adds what the tables do not have:
who held the ball, where players were, how fast they moved in this clip.

**Court mapping instead of pixel maths.** Speed and distance are only meaningful in metres. The court is
a known rectangle with painted lines, so a few frames were calibrated by hand and the rest are matched to
them with SIFT features, then nudged so the projected lines sit on the painted ones. A frame counts as
mapped only if enough features match and the lines line up; otherwise it is skipped, not guessed.

**Smoothing before differencing.** Raw foot positions jitter by tens of centimetres; differencing that
gives absurd speeds. Positions are smoothed (Savitzky-Golay, 0.5 s), tiny speeds count as standing, and
speeds above a physical limit are discarded.

**Teams from shirt colour.** K-means on each track's shirt colour gives light kit, dark kit and a
referee group. Which kit is which team was read off the pictures and is set in `config.yaml`.

**Ball by self-training.** The generic detector sees this ball only at very low confidence. Its confident,
orange, non-static, near-a-player detections became training labels for a small ball-only model, and a
path-through-time search picks the true ball among candidates (ignoring static look-alikes). The model's
validation score only measures agreement with those self-made labels; the honest check is by eye
(`results/ball_model/visual_check.json`).

**Possession rules are written down.** A holder is the player whose box contains the ball; frames where
the ball sits convincingly inside players of both teams get no holder. A pass is a change between two
teammates; an interception is a change to the other team away from the baskets; a steal answered back
within 1.5 s is dropped as noise. The definitions are in the docstring of `src/bball/analytics/possession.py`.

**The ball is drawn at the holder, not projected.** A ball in the air cannot be located on the floor from
one camera, so the replay shows it next to the player the analysis says holds it.

## 3. What went wrong along the way (and what it taught)

* The first interception count was far too high: overlapping opposing players flipped the "holder" for
  a frame or two. Looking at a contact sheet of the detected events (not just the totals) exposed it, and
  the fix was to refuse to name a holder when the ball is shared and to cancel instant flip-flops.
* Filtering people by "do they move?" threw away most real players who stood still in the half court.
  Replaced by "is the person inside the court lines, and either well inside or fast".
* The ball detector first locked onto painted dashes and a trophy icon in the score bar; static-candidate
  removal and an overlay mask fixed that.
* NBA.com's advanced table for some older games has a ghost duplicate player row (same id, no minutes); the player page now
  keeps only rows with minutes.
* The top navigation links were unclickable because Streamlit's transparent header sat on top of them,
  which was found by asking the browser which element is topmost at each link, not by guessing.

## 4. Questions I should be able to answer

* **Is the video analysis accurate?** Unknown in the strict sense: there are no ground-truth labels. The
  README states the by-eye checks and compares with the official steals and turnovers for the quarter.
  The video interceptions are expected to be over-counted and passes under-counted because the ball is
  not visible in a large share of frames.
* **Why are some stats missing?** NBA.com does not publish PER, win shares, drives, time of possession,
  potential assists, catch-and-shoot splits or contested-rebound % for a single game; the site lists them
  instead of inventing values.
* **Why does the hosted site show a "demo sample"?** NBA.com's stats servers often ignore requests from
  public cloud hosts, so a hosted copy cannot fetch live data. The site detects that (one short probe, then
  a 10-minute memory) and falls back to a bundled sample of every Finals game of 2016-2022, with a visible
  notice. Run locally for all playoff games.
* **Is it allowed?** The stats are NBA.com's and covered by its terms (not reviewed in full), and a small
  sample of them is bundled in the repository for the hosted demo; the original
  clip is not in the repository; the annotated output (one still in the README, the full demo as a release
  asset) is shared for illustration and the footage remains the rights holder's, so it is removed on request. Ultralytics YOLO is AGPL-3.0. The background photo
  was supplied by the repository owner as free to use and could not be verified.
* **What would improve it most?** A second round of ball training with human-checked labels, and a
  jersey-number reader so a player can be followed across camera cuts.
