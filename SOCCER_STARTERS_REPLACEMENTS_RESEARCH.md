# Soccer starter importance, absences and replacements — experimental (V17)

## Objective
Research the user's question: **does a key starter's absence or replacement help explain who wins a football match?** Use only timestamped pre-kickoff injury and squad data, plus archived results observed in earlier completed matches. No player-props recommendations, no betting execution and NO automatic adjustment of champion odds.

## Coverage and sources
- ESPN soccer event-level completed player stats and pregame rosters, already archived in private `kalshi-ai-data`.
- API-Football injury statuses, linked to an ESPN fixture **only after both team names and kickoff match uniquely**. The provider-specific athlete ID is never assumed to equal an ESPN ID. Injured names are mapped to prior ESPN athletes only by a single exact normalized name within a proven team. Ambiguous/missing mappings remain unknown.
- API-Football `/fixtures/lineups` is queried **only for up to four uniquely matched fixtures within the next two hours** in the existing twice-daily collection job. Two complete distinct XIs (11 each) captured at least 5 minutes before kickoff are required before the record is considered a verified pregame lineup. This does **not** imply it will always be available: lineups commonly appear 20–40 minutes before kickoff, and twice-daily snapshots will miss many. No extra cron schedule or paid services enabled.

## Role and replacement evidence
`kalshi_external/soccer_starter_impact.py` builds a 0–100 **historical minutes/participation index**, not a win-probability or causal impact. The index requires at least four prior team games and three recorded minute appearances for the specific athlete, from the same competition, with every game's boxscore captured before the relevant forecast snapshot. `None` means insufficient evidence.

For a uniquely matched player marked `Missing Fixture`, the report shows the historical role index and up to three same-position **possible cover players**, ranked by the same evidence. Without a verified starting XI they are clearly *unconfirmed candidates*. Even with a confirmed XI, exact-name mapping must be unambiguous before the alternative is called an observed starter. Questionable players remain uncertain, never automatically absent.

Teams with missing, ambiguous or unavailable history retain an unknown role-burden signal rather than assigning a fictitious percentage. A positive `home_minus_away_absence_exposure` indicates more identified absences on the away side, **not** a percentage point advantage.

## Prospective storage
- `soccer_confirmed_lineups.jsonl` and `soccer_starter_impact.jsonl` are added to private `kalshi-ai-data`, restored and persisted every regular external-research run.
- `soccer_impact_collect.py` pairs context with the champion's 1X2 probability only when its exact ESPN event, competition, kickoff and forecast creation timestamp are verified before the player observation. It never modifies those predictions.
- A quality gate (`soccer_impact_audit.py`) rejects unconfirmed or post-kickoff lineups, future model snapshots, corrupt role indices or duplicate events.
- Aggregate-only files in `reports/external_learning/soccer_impact/` show observed coverage, never an unsupported performance guarantee.

## Evaluation gate
On the existing Monday validation run, `soccer_impact_shadow.py` joins private prospective impact observations to confirmed 90-minute `H/D/A` match results. It uses a chronological train/test split by kickoff date, requires **at least 120 training and 50 untouched holdout matches over 20+ test dates**, compares bounded log-odds adjustment weights fitted on past matches, and calculates paired logloss, Brier and cluster-bootstrap uncertainty. A candidate is only marked `research_only` when the held-out logloss 95% lower CI and Brier gain are both positive.

**No candidate is promoted or used for wagering even if the test passes.** True robustness needs multiple seasons, confidence/coverage analysis, close-to-kickoff data and independent prospective replication. Small samples cannot justify a claim that a specific player changes a win chance by X%.

## Source documentation
- API-Football lineups: https://www.api-football.com/documentation-v3
- API-Football fixture-player statistics: https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide