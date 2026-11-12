# Multisport source challenger training (research only)

Kalshi AI now evaluates licensed-source prediction mixtures **separately by sport**: international teams, club soccer, NBA, NFL. It uses prospective pregame snapshots archived privately, joined to final results by `(sport, event_id)` to prevent cross-sport contamination.

Weekly workflow restores the full `kalshi-ai-data` private archive using existing `DATA_REPO_TOKEN` before any challenger comparison. Outcomes come from the verified ESPN result lookup (no future events), saved to aggregate `reports/external_learning/resolved_outcomes.csv`. The private archive is never published to the source repository.

The research learner fits a blend weight from the earlier 70% of complete dates and evaluates it on the later 30% of resolved events. Each source is scored separately for each sport. Logloss is the selection objective, and the results include an uncertainty estimate and a clear RESEARCH_ONLY status. No blending parameter enters the champion model.

Only `source_kind=market` and explicitly authorized `source_kind=analyst` qualify for weight exploration; Skorbot remains `research_only`. NOAA meteorological observations are separate features, never passed to the probability learner.

No credible improvement claim is possible until sufficient new results have settled and the model's as-of probabilities are preserved. NFL/NBA `h2h` uses 2 outcomes; soccer 1X2 has 3 outcomes. Player props are not supported yet. No automated bets or model promotion.