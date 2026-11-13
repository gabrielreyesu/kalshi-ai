# Viajes y hora del partido (NBA/NFL) — protocolo pre-registrado

**Fijado el 13-nov-2026, antes de ver resultados.** Investigación: no cambia picks ni probabilidades.

## Variables (positivo favorece al local; solo información conocida antes del inicio)
- `travel`: (km del visitante desde la sede de su partido anterior − km del local) / 1000.
- `tz`: husos horarios cruzados por el visitante desde su partido anterior − los cruzados por el local (valor absoluto).
- `clock`: huso de la sede − huso de la ciudad del visitante (positivo = el visitante viene del oeste).
- Sedes: estadio del local con coordenadas y zona horaria fijas (incluye traslados de Rams, Chargers y Raiders).
  Partidos marcados neutrales (Londres, México, etc.): variables en 0.

## Evaluación
- Variantes: **A** `travel`; **B** `travel` + `tz` + `clock`.
- Frente al modelo: walk-forward por temporada (`kalshi_lineup.evaluate`). NBA: desarrollo ≤ 2024, confirmación ≥ 2025.
  NFL: desarrollo ≤ 2023, confirmación ≥ 2024. Éxito: IC95% de mejora de log loss > 0, Brier no peor, ECE no peor > 0,005.
- **Frente al mercado** (la prueba que decide si entra al pick final): línea de cierre de las casas (ESPN) + variante
  elegida, walk-forward; éxito si la mejora sobre el mercado recalibrado tiene IC95% > 0.

## Resultados (13-nov-2026) — NO_SUPERA_CRITERIO en ambos
| Liga | Confirmación | Variante elegida | Mejora vs modelo [IC95%] | Mejora vs mercado [IC95%] |
|---|---|---|---|---|
| NBA | 2.817 partidos (≥ 2025) | B | −0,0002 [−0,0010; +0,0006] | −0,0004 [−0,0008; −0,0000] |
| NFL | 707 partidos (≥ 2024) | B | −0,0005 [−0,0046; +0,0037] | +0,0005 [−0,0035; +0,0046] |

Lectura: en NBA el modelo ya incluye descanso y partidos consecutivos (lo que captura el cansancio) y el mercado ya
valora los viajes; en NFL la semana entre partidos diluye el efecto. Descartado; no entra al pick final.
