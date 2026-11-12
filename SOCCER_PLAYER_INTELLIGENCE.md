# Kalshi AI — Soccer player availability and opponent performance (research only)

## Coverage
Data collection covers ESPN scoreboard and limited completed match summaries for Champions League, Europa League, Conference League, Premier League, LaLiga, Serie A, Bundesliga, Ligue 1, UEFA Nations League and UEFA World Cup qualifiers. These span **club and national-team football**. The existing model's ESPN IDs are not the same as IDs from API-Football or football-data.org.

## Signals
- Completed match statistics when explicitly returned by ESPN: athlete/team IDs, minutes, goals, assists, shots, shots on target and saves. A game with no per-player stats is **not counted** as a historical player-stat game.
- Pregame roster snapshots only when observed before kickoff; unless a provider explicitly verifies official lineups, `lineup_confirmed=false`. A listing or starting XI *hint* must not be presented as a confirmed XI.
- Optional API-Football player injury/suspension reports via existing `API_FOOTBALL_KEY` GitHub Actions secret. Official API docs distinguish `Missing Fixture` from `Questionable`. Without configured credentials or permission on the user’s plan, that coverage is unknown and **not assumed healthy**.
- API-Football and ESPN IDs are stored separately. A report can be attached to an ESPN fixture only after uniquely matching **both teams and UTC kickoff**, within 60 minutes, using actual API-Football fixture metadata. Provider player ID is not treated as an ESPN athlete ID.
- For each upcoming fixture, build player-versus-opponent historical splits from games already captured and completed prior to the snapshot. Require two previous meetings and shrink the opponent mean toward the recent overall mean `n/(n+6)` to avoid exaggerated claims about small samples.

## Automation
- `.github/workflows/external_learning.yml` runs twice daily and now invokes `kalshi_external.soccer_player_collect` and `soccer_player_audit`.
- Raw individual-level records are stored only in private `gabrielreyesu/kalshi-ai-data` as `external_signals/soccer_player_games.jsonl`, `soccer_lineups.jsonl`, `soccer_injuries.jsonl`, `soccer_previews.jsonl`.
- Public workflow artifacts contain only **aggregate** coverage and quality audit counts (`reports/external_learning/soccer_players/`), not private player-level data.
- Limits: at most 12 per-match summary queries in a cycle, 10 fixed competitions across a 3-day window; optional API-Football injuries and fixture metadata once per UTC date when a key is configured. ESPN’s public site API may have incomplete feeds or change schema; errors result in low coverage, not fabricated records.

## Limits and next scientific step
- This is **not a player-props system** and **does not adjust production predictions** or change bets. Data on absences and prior opponent splits are descriptive; they do not demonstrate a causal win-probability effect without controlling teammate replacements, form, position, quality and strategy.
- Lineups typically appear close to kickoff; twice-daily snapshots can miss them. Dedicated authorized near-kickoff observation may be added after source coverage and provider terms have been verified.
- Explicitly test whether availability, expected minutes, starting XI, role and opponent interaction improve a prospective walk-forward evaluation of 1X2 vs the existing champion + contemporaneous de-vigged consensus. Avoid leakage from postgame player stats and retrospective lineup reconstruction.

## Provider documentation
- https://www.api-football.com/documentation-v3 (injuries, lineups and per-fixture player stats; coverage varies by league and subscription)
- https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions/scoreboard (public ESPN site feed already used by Kalshi AI; availability and usage terms may change)
## Data-identity policy
ESPN team IDs are not assumed globally unique between competitions: player-versus-opponent historical comparisons currently use the **same ESPN competition code** only, even if that reduces coverage. API-Football injury lists and their original fixture metadata are preserved together so an earlier same-day report remains usable in the next snapshot, but only when the entire match uniquely matches ESPN by both names and kickoff. Only the newest report for each fixture date is used and the report must have been captured within 36 hours. Never attach a vendor player ID directly to an ESPN ID.
