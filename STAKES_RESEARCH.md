# ¿Qué se juega cada equipo? (clasificación y motivación) — protocolo pre-registrado

**Fijado el 13-nov-2026, antes de ver resultados.** Investigación: no cambia picks ni probabilidades.

## Hipótesis
Un equipo que ya no puede cambiar su destino en la tabla (título, Europa o descenso decididos) rinde menos que su nivel
habitual frente a un rival que aún se juega algo. Los modelos actuales solo ven resultados, no la tabla.

## Variables (ligas: Inglaterra, España, Italia, Alemania, Francia, Portugal y Países Bajos)
- Clasificación reconstruida solo con partidos de liga jugados **antes** de la fecha (3/1/0 puntos).
- Rango posible de posición final ignorando desempates: mejor = 1 + rivales inalcanzables; peor = 1 + rivales que aún
  pueden igualarlo o superarlo.
- Fronteras: título (1.º), Europa (top 4; Francia top 3; Portugal y Países Bajos top 2), descenso (últimos 3).
- "Algo en juego" = el rango posible cruza alguna frontera. Variantes: **A** `stake_diff` (local − visitante);
  **B** `nostake_h` y `nostake_a` por separado.

## Evaluación
Motor de clubes de `kalshi_intl` (Elo + Poisson), walk-forward por temporada, evaluado en partidos de liga (1X2 a 90 min).
Desarrollo 2020/21–2023/24 elige una variante; confirmación ≥ 2024/25 una sola vez. Éxito: IC95% de mejora de
log loss > 0 (bootstrap por fecha), Brier no peor, ECE no peor en más de 0,005. Descriptivo: solo partidos con algún
equipo sin nada en juego (son los que la variable puede cambiar; suelen ser las últimas jornadas).

## Siguiente (si funciona)
Fase de liga de Champions/Europa League (última jornada) y NBA/NFL (equipos ya clasificados o eliminados), con el
mismo protocolo. En NBA/NFL parte del efecto ya lo captan las variables de alineación (titulares que descansan).

## Resultados (13-nov-2026)
### Ligas — NO_SUPERA_CRITERIO
Confirmación 2024/25–2026/27, 5.437 partidos: variante A mejora log loss +0,0003 [−0,0004; +0,0010]; Brier +0,0001.
Solo los 403 partidos con algún equipo sin nada en juego: +0,0035 [−0,0056; +0,0131]. Dirección esperada, efecto
demasiado pequeño para distinguirlo del azar. Desarrollo: A +0,0005 [−0,0001; +0,0010], B +0,0002.

### Nations League + eliminatorias UEFA (grupos inferidos de los emparejamientos) — NO_SUPERA_CRITERIO
Confirmación: 495 partidos, −0,0015 [−0,0040; +0,0004]; 49 partidos con algún equipo sin nada en juego: −0,015
[−0,036; +0,005]. Las selecciones sin nada en juego suelen ser equipos débiles cuyo rating ya lo refleja.

### Decisión
No se incorpora al modelo. La página muestra un **aviso informativo** (⚑) cuando un equipo ya no se juega nada en su
grupo o tabla (`python -m kalshi_lineup.stakes_now`, en soccer-daily), sin cambiar la probabilidad. Repetir el
backtest al final de cada temporada de ligas (más partidos con equipos sin nada en juego) y extenderlo a la fase de
liga de Champions/Europa League y a NBA/NFL.

Verificado el 13-nov-2026 con datos reales de ESPN: 14 grupos de la Nations League 2026 (12 de 4 equipos, 2 de 3); 4 partidos de la jornada del 15-nov con un equipo sin nada en juego (Letonia, Montenegro, San Marino, Turquía).
