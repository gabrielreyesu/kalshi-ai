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

## Resultado NBA (ejecución 43432220478, 13-nov-2026)
Cobertura: 14.163 de 14.167 partidos finales con boxscore (99,97%).

**Desarrollo (2019–2024, 7.413 partidos):** las 4 variantes mejoran con IC95% > 0. Elegida: **D (disponibles + fuerza absoluta)**,
mejora de logloss 0,0089 [0,0064; 0,0114].

**Confirmación (temporadas ≥ 2025, 2.817 partidos, evaluada una vez):**
| | Acierto | Logloss | Brier | ECE |
|---|---|---|---|---|
| nba_v1 | 67,02% | 0,60295 | 0,20818 | 0,0201 |
| nba_v1 + alineación (D) | 67,41% | 0,59319 | 0,20417 | 0,0215 |
| Mejora (IC95% por fecha) | | **0,0098 [0,0054; 0,0142]** | **0,0040 [0,0021; 0,0060]** | +0,0014 (límite 0,005) |

**Veredicto: SUPERA_CRITERIO_SOLO_SOMBRA.** Mejora en las 9 temporadas, sin excepción.

**Frente al mercado de cierre** (10.229 partidos con línea): modelo 65,1% / 0,6215 → con alineación 66,1% / 0,6123;
mercado 67,7% / 0,5982. Cierra ~40% de la distancia, pero el mercado sigue mejor.
**Alineación sobre el mercado:** sin aporte (−0,0004 [−0,0008; 0,0000]), como se esperaba: el cierre ya incorpora las bajas.

Lectura: la variable mejora de forma robusta el modelo propio, que es la probabilidad de respaldo cuando no hay
mercado y la base para medir si recalculamos antes de que el precio se ajuste. No justifica por sí sola apostar contra
el cierre. Siguiente paso: sombra prospectiva con el informe de lesiones real capturado a T-60 min (no el boxscore).

## Resultado NFL: quarterback titular (ejecución 43433437322)
Cobertura: 3.165 de 3.165 partidos finales con boxscore. Desarrollo 2019–2023; confirmación ≥ 2024 (una vez).

**Desarrollo:** A (solo cambio de QB) mejora 0,0052 [0,0006; 0,0098]; **B (cambio + nivel del titular)** mejora 0,0103 [0,0039; 0,0165]. Elegida: B.

**Confirmación (707 partidos, temporadas 2024–2026):**
| | Acierto | Logloss | Brier | ECE |
|---|---|---|---|---|
| nfl_v1 | 65,91% | 0,62128 | 0,21620 | 0,0432 |
| nfl_v1 + QB (B) | 66,90% | 0,60796 | 0,21007 | 0,0338 |
| Mejora (IC95% por fecha) | | **0,0133 [0,0037; 0,0237]** | **0,0061 [0,0019; 0,0108]** | mejor |

**Veredicto: SUPERA_CRITERIO_SOLO_SOMBRA.** B mejora en las 8 temporadas. En los 712 partidos con cambio de QB, la mejora
de logloss es 0,0295 [0,0156; 0,0435] (descriptivo).

**Frente al mercado de cierre** (2.096 partidos): modelo 0,6354 → con QB 0,6241; mercado 0,6093. Cierra ~43% de la distancia.
**QB sobre el mercado:** sin aporte (−0,0005 [−0,0029; 0,0021]).

Cautela: la muestra de confirmación de NFL es pequeña (707 partidos, IC ancho). La regla del titular usa el boxscore
final; en producción debe venir del depth chart/inactivos oficiales con hora de captura.

## Fútbol: once titular (protocolo pre-registrado, antes de ver resultados)
- Datos: resúmenes de ESPN (`rosters`, marca `starter`) desde 2018-07. Clubes: Champions, Europa League, Conference y
  ligas de Inglaterra, España, Italia, Alemania, Francia, Portugal y Países Bajos. Selecciones: todas las competiciones del motor.
- Solo se usa quién es titular (publicado ~60 min antes). Goles y asistencias del partido solo actualizan después.
- Variantes: **A** continuidad del once (peso de titulares habituales con decaimiento 0,85 por partido);
  **B** A + goles/asistencias por titularidad del once de hoy frente a sus últimos 10 onces.
- Equipos con menos de 3 onces previos: variables en 0 (neutras).
- Evaluación: `kalshi_intl.model.walk_forward` (1X2 a 90 min, competiciones objetivo). Desarrollo 2020/21–2023/24;
  confirmación ≥ 2024/25 una sola vez. Mismo criterio de éxito que NBA (logloss IC95% > 0, Brier multiclase no peor,
  ECE no peor en más de 0,005).

## Pruebas prospectivas en sombra (datos publicados ANTES del partido)
| Deporte | Workflow | Dato previo al partido | Variante | Archivo |
|---|---|---|---|---|
| NBA | `nba-lineup-shadow` (9×/día) | Informe de lesiones ESPN (Out/Doubtful/IR fuera) sobre la rotación reciente | D | `predictions/nba_lineup_shadow*.{csv,json}` |
| NFL | `nfl-qb-shadow` (6×/día) | Depth chart ESPN: primer QB no Out/Doubtful/IR | B | `predictions/nfl_qb_shadow*.{csv,json}` |
| Copas europeas | `soccer-xi-shadow` (cada 15 min, mar–jue 15–20 UTC) | Once oficial ESPN, 5–80 min antes del inicio | A | `predictions/soccer_xi_shadow*.{csv,json}` |

Cada uno puntúa con la última foto anterior al inicio (log loss y acierto, IC95% por fecha). Regla de activación común:
n ≥ 200, IC95% de mejora > 0 y autorización explícita. Copas: la primera vez ejecutar `soccer-xi-shadow` a mano con
`warm = true` para llenar la caché del historial de onces (~2 h); después cada ejecución sin partido dura segundos.
Verificado con datos reales el 13-nov-2026: NFL detectó 13 partidos de la semana 11 y 5 cambios de QB; el lector de
onces capturó onces oficiales 7 y 22 minutos antes de partidos en vivo.
