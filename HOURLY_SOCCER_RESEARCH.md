# Kalshi AI: hourly football research monitoring

## New GitHub Actions cadence
The external research workflow uses **two complementary GitHub cron expressions**, together covering every hour at minute 35 UTC. The intended 01:35 and 13:35 UTC schedules always select the historical/full profile, while the other 22 scheduled hours use `--mode hourly`. Profile selection uses the matched GitHub cron expression rather than the runner's start clock: GitHub has delayed real scheduled runs, and using wall-clock time could accidentally skip full research. Push/manual dispatch stays full. GitHub's scheduled delivery is best effort; the hourly cadence is a target, not a guarantee at exactly minute 35.

## Fast hourly profile
- Only rescan ESPN football competitions with previously detected fixtures due in the next 72 hours. The two full scans discover new competitions and keep the upcoming window populated. Unlike full mode, hourly never requests monthly historical scoreboards and never requests completed match summary details.
- ESPN scoreboard lookups cover today and the next two UTC dates, filtered to the relevant competitions. Full historical acquisition stays twice daily rather than 24 times daily.
- API-Football injury reports are refreshed no more frequently than **every four hours per target fixture date**, matching the provider's published update interval. The original API-Football fixture index is reused when valid and filters out any game that already began. A report is considered current for the event only while its actual capture timestamp is at most six hours old; no injury is inferred to be absent if the provider feed fails.
- Up to **two** API-Football `fixtures/lineups` calls per hour for exact ESPN/provider fixture matches starting between 15 and 75 minutes later; games with a successfully archived confirmed 11v11 XI are never queried again. A complete XI must be captured at least five minutes **before kickoff**; lineups fetched after the match are rejected.
- The full and hourly profiles both build timestamped soccer starter/replacement research, run strict data audits, and save verified raw player data only in private `kalshi-ai-data`. The public workflow logs/artifacts contain aggregate status, including collection mode, number of provider requests this run, and number of confirmed XIs.

## API quota policy
API-Football currently lists 100 calls/day for its free plan and 7,500/day for Pro. This workflow is designed to keep a conservative maximum: two injury requests per four-hour window (about 12/day), up to two lineup requests per hourly run (48/day) only where close match fixtures exist, plus a few cached fixture-index lookups. Other connected workflows have their own request use; cumulative quota is not guaranteed under extreme manual re-runs. A provider error or missing secret must **not** be interpreted as an empty injury list or an official starting XI.

## What happens to picks?
The football **data freshness** improves to hourly, especially for pregame XI context. This is research-only: production 1X2 probabilities still come from the separate soccer model workflow. Refreshing an injury list does **not** imply that existing model probabilities have been recalibrated or have improved accuracy. The shadow evaluator needs enough settled cases to test changes in log loss/Brier before allowing any player-availability feature into a champion model. Market prices can change between snapshots; stale source timestamps must remain visible.

## Documentation
- API-Football: https://www.api-football.com/documentation-v3 (lineups often published 20–40 minutes before kickoff; injuries updated ~4h).
- Pricing: https://www.api-football.com/pricing/ (free tier and request quotas).
- GitHub Actions scheduled workflows: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
## Archive growth controls
The collector compares each upcoming match's *material player research* with the most recent privately saved snapshot, ignoring only the timestamp of the check itself. If all player facts, source injury timestamp and official XI timestamp are unchanged, it reports a successful hourly inspection but does **not** rearchive another full player-by-player blob. A refreshed report (even with unchanged injury statuses) or newly confirmed XI **is** recorded with a new observation timestamp. This retains a trustworthy pre-kickoff audit trail without filling the 100 MB JSONL safety caps in weeks of idle hourly monitoring.
