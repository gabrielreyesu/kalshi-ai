# Kalshi AI — Player intelligence and opponent matchups (research only)

The existing `kalshi_nba.injuries` collects injury lists once per day and records equal counts regardless of athlete role. It does not adjust model probabilities. The new pipeline adds timestamps, IDs and research-only athlete performance data.

## Acquisition
- ESPN site NBA/NFL public injury JSON snapshots twice daily; record athlete/team IDs, position, status and reported/capture time. Absence from the report **does not imply fit or playing**.
- A bounded sample of **completed** ESPN boxscores: NBA minutes, points, rebounds, assists; NFL passing, receiving and rushing yards plus receptions. Never use a result before the timestamp it was collected.
- For upcoming NBA/NFL fixtures within 48 hours, construct team-specific player contexts, preserving latest pregame injury snapshot, prior games and opponent IDs. H2H summaries require at least two meetings and shrink to the player's recent mean using n/(n+6). These are descriptive stats, not causal estimates.
- Full individual-level data are archived only in private `gabrielreyesu/kalshi-ai-data`, files `external_signals/player_injuries.jsonl`, `player_boxscores.jsonl`, `player_previews.jsonl`. GitHub artifacts contain **aggregate coverage and archive health**, not athlete details.
- The same workflow that already takes external-odds snapshots handles acquisition and restoration. No extra scheduled runs, provider subscriptions or secrets required; endpoint requests are bounded (up to 6 completed games per sport per run, last 3 days).

## Controls and limitations
- Official league injury reports (NBA official page, NFL.com) remain useful for manual cross-check; ESPN endpoints are public/unofficial, may be delayed or incomplete, and do not establish a player as a confirmed starter.
- In the context builder, injury snapshots older than 36 hours are ignored, reports dated after collection discarded and only prior completed games available before the forecast can be used. The player research audit rejects malformed source IDs, duplicate snapshots and invalid values before private persistence.
- **No automatic adjustment to win probability yet.** A missed player might affect the probability greatly, but quantifying it needs starter/backup information, expected workload and opponent strength with prospective chronological evaluation. The data are not betting advice or direct player props.
- NBA/NFL first. Soccer requires an independent authorized canonical player and pre-match lineup source.

## Evaluation plan
Build enough pre-game snapshots, then test whether adding injury availability and player-strength features to the champion improves independent holdout accuracy, Brier score and log loss, benchmarked to same-game market odds. Prefer partial pooling, explicit uncertainty, no automatic model promotion, and no look-ahead leakage.

## NFL live feed fallback

ESPN's NFL injury endpoint can sometimes fail or respond with an unexpected schema. The system now retries safe HTTP failures and, when the live NFL feed cannot be used, reads the **same-day** `predictions/nfl_injuries.csv` that the existing daily workflow already captured. The fallback is **not up-to-the-minute verified**; it carries `feed_origin=same_day_cached_espn_csv`, canonical ID is explicitly false, and no lineup is presumed confirmed.

When the fallback lacks athlete IDs, player research links by **unique exact team+name** only if a prior player boxscore has one unambiguous ESPN athlete ID. If names are duplicated or mismatched, no join is made and no role-impact claim is inferred. The private audit still runs before archival; the model in production is unchanged.
