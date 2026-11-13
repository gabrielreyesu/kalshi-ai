# NBA: ratings ajustados por alineación (investigación, P1)

**Estado:** investigación en modo sombra. **No cambia** `kalshi_nba`, `predictions/` ni ninguna probabilidad publicada.

## Por qué
En 10.225 partidos fuera de muestra, el mercado acierta 67,6% frente a 65,1% de `nba_v1` (logloss 0,598 frente a 0,622).
`nba_v1` solo ve resultados: no sabe quién juega. Las lesiones archivadas desde noviembre de 2026 tardarían meses en
permitir una validación. Este módulo reconstruye la disponibilidad **histórica** desde los boxscores de ESPN y la
valida ahora, temporada a temporada.

## Qué información se usa y cuándo se conoce
| Dato del partido | ¿Se usa para ese partido? | Cuándo se conoce |
|---|---|---|
| Titulares | Sí (variante B) | ~30 min antes del salto inicial |
| Disponibles (jugó o "DNP - decisión del técnico") | Sí (variantes A, C, D) | Lista de inactivos, antes del partido |
| Minutos, puntos, +/- del propio partido | **No**. Solo para actualizar después | Después del partido |

Limitación declarada: "disponible" se infiere del boxscore final. Una baja de último minuto en el calentamiento puede
aparecer como inactiva aunque un informe de lesiones a T-60 no la tuviera. Por eso la aplicación en producción debe
usar el informe de lesiones / la alineación real con su hora de captura, y validarse de forma prospectiva.

## Valor de jugador (as-of)
- Game Score (Hollinger) por minuto, con memoria de vida media de 120 días y una previa de 300 minutos hacia nivel de reemplazo
  (75% de la media de la liga). Variante C añade +/- por minuto con previa de 1000 minutos.
- Minutos esperados: media exponencial (α=0,3) de los minutos cuando estuvo disponible.
- Fuerza del equipo hoy = Σ valor × minutos asignados (240 minutos repartidos según minutos esperados, máximo 42 por jugador).
- Variable = fuerza de hoy − media de la fuerza de sus 10 alineaciones anteriores (recalculada con valores actuales),
  local menos visitante.

## Protocolo pre-registrado (fijado antes de ver resultados reales)
1. Walk-forward por temporada (`kalshi_nba.backtest.walk_forward`): cada temporada se predice con un modelo entrenado
   solo con temporadas anteriores. Mismos partidos para base y variantes.
2. **Desarrollo:** temporadas con año de cierre 2019–2024. Se comparan 4 variantes fijas (A disponibles, B titulares,
   C disponibles + +/-, D disponibles + fuerza absoluta) y se elige UNA por mayor mejora media de logloss.
3. **Confirmación:** temporadas ≥ 2025, evaluadas una sola vez con la variante elegida.
4. **Criterio de éxito:** IC95% de la mejora de logloss (bootstrap por fecha) > 0 en confirmación, Brier no peor, y
   ECE (calibración, 10 tramos) no peor en más de 0,005. Si no se cumple, se descarta.
5. Además se informa (sin decidir nada con ello): modelo vs mercado de cierre en el mismo subconjunto, y si mercado +
   alineación mejora al mercado recalibrado. Se espera que no, porque el cierre ya incorpora las alineaciones.

Aunque supere el criterio, el siguiente paso es **solo sombra prospectiva** con el informe de lesiones real a T-60 min,
y la activación requiere autorización explícita.

## Cómo correrlo
- GitHub Actions: workflow `NBA lineup-adjusted ratings research` (se ejecuta al subir cambios a ramas `research/**`
  o a mano). Informe en el resumen de la ejecución y en el artefacto `nba-lineup-research-*`.
- Local: `python -m kalshi_lineup.cli backtest` (la primera vez descarga ~13 mil boxscores; caché en `data/nba/box/`).

## Pruebas
`tests/test_lineup.py`: lectura del esquema de ESPN (por `keys` y por `labels`), regla de disponibilidad, fallos de red
no guardados en caché, ausencia de fugas (cambiar estadísticas desde una fecha no altera variables de esa fecha ni de
fechas anteriores), partido sin boxscore = variables neutras, recuperación de señal sintética y control nulo (sin señal
real, no aparece una mejora espuria).
