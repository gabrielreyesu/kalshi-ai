# NFL: distribución real de márgenes para los hándicaps — protocolo pre-registrado

**Fijado el 13-nov-2026, antes de ver resultados.**
- Hoy: P(local gana por más de x) = curva normal (σ 12,7) centrada en −spread de las casas.
- Alternativa: distribución empírica del residuo (margen real − (−spread)) de temporadas anteriores.
  **A** sin tramos; **B** por tramo de |spread| (≤3, 3,5–7, >7).
- Evaluación: para cada partido, líneas desde −14,5 hasta +14,5 alrededor del spread (30 líneas); log loss medio por
  partido. Walk-forward por temporada (prueba ≥ 2019). Desarrollo 2019–2022 elige A o B; confirmación ≥ 2023 una vez.
  Éxito: IC95% de mejora > 0 (bootstrap por fecha). Si pasa, sustituye a la normal en la probabilidad de las casas
  (`p_book`) de los hándicaps de NFL, que alimenta los picks de hándicap de la página.

**Corrección de implementación (antes del resultado final):** la primera ejecución usaba líneas = spread ± k, que caen
en números enteros cuando el spread es x,5 (empates mal contados). Se corrigió a líneas siempre x,5 (como en Kalshi),
que es lo que pretendía el protocolo. Primera ejecución (con el fallo): A +0,0004 [−0,0007; +0,0016], no concluyente.

## Resultado (13-nov-2026) — NO_SUPERA_CRITERIO
Desarrollo 2019–2022: A +0,0005 [−0,0009; +0,0018]; B −0,0005. Elegida A. Confirmación ≥ 2023 (991 partidos):
log loss normal 0,55592 vs empírico 0,55579, mejora +0,0001 [−0,0010; +0,0012]; líneas a ±7,5 del spread: +0,0005
[−0,0016; +0,0026]. La curva normal ya está bien calibrada en promedio; se mantiene. No se guarda tabla empírica.
