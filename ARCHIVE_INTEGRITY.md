# External archive data-quality gates

Goal: preserve valid odds and NOAA weather observations in the durable private repository, and stop data corruption from spreading unnoticed.

On every external source workflow, `python -m kalshi_external.archive_audit` reviews the restored private research archive plus new snapshots **before** committing. It validates (1) structured JSON, (2) nonduplicated event/source/capture identifiers, (3) forecast capture and publication before kickoff, (4) bookmaker and baseline probability normalization, and (5) plausible weather units. Invalid data cause an explicit CI failure and block the private archive commit (the workflow artifact remains available for troubleshooting). Missing/empty archive files are allowed when starting fresh.

Only aggregate results go to `reports/external_learning/archive_health.json`, not raw paid-provider data. Checks are research quality controls; they do not introduce information into production models and do not place bets.

Observed private snapshot history before implementing audit (2026-11-12): **203** odds observations for **29** distinct sport/events, plus **3** NOAA weather observations, no duplicates or captures after kickoff. Those are observations, not wins.