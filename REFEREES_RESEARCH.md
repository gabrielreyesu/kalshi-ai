# Árbitros en la NBA — protocolo pre-registrado

**Fijado el 13-nov-2026, antes de ver resultados.** Investigación: no cambia picks ni probabilidades.
- Árbitros de cada partido: resumen de ESPN (`gameInfo.officials`). Se publican la mañana del partido.
- Tendencia por árbitro con partidos **anteriores**: media encogida hacia 0 (k = 40 partidos) del residuo de totales
  (total real − línea de cierre) y del residuo de margen (margen local + spread). Variable = media del trío.
- **Totales:** residuo ~ tendencia del trío, walk-forward por temporada (≥ 2019). Confirmación: temporadas ≥ 2025.
  Éxito: reducción del error cuadrático frente a la línea con IC95% > 0.
- **Ganador:** logit(probabilidad de las casas) + tendencia de margen del trío. Confirmación ≥ 2025; éxito IC95% > 0.

## Resultado (13-nov-2026)
14.074 partidos con árbitros. **Totales — no concluyente (prometedor):** reducción del error cuadrático en desarrollo
+0,33 [−0,05; +0,72] y en confirmación ≥ 2025 (2.812 partidos) +0,50 [−0,30; +1,31] puntos²; misma dirección en
ambos tramos pero IC incluye 0. **Ganador — sin efecto:** +0,0001 [−0,0001; +0,0003]. No entra aún al pick; repetir
con más temporadas (y probar el efecto por árbitro principal en lugar de la media del trío).

## NFL (mismo protocolo; confirmación ≥ 2024) — no concluyente
2.732 partidos con árbitros. Totales: desarrollo −0,22 [−0,61; +0,17], confirmación (682) +0,58 [−0,01; +1,16]
puntos²: signo distinto entre tramos, no fiable. Ganador: −0,0001 [−0,0005; +0,0003]. No entra al pick.

## Fútbol — protocolo (fijado antes de ver resultados)
Árbitro principal (ESPN `gameInfo.officials`, orden 1). Tendencia con partidos anteriores, encogida (k = 25): exceso de
"más de 2,5" sobre la probabilidad de las casas y exceso de victorias locales sobre las casas. Ligas grandes + copas
europeas desde 2019/20. logit(casas) + tendencia, walk-forward por temporada (prueba ≥ 2021/22), confirmación ≥ 2024/25.
Éxito: IC95% > 0 en confirmación.
