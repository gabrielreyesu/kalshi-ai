# Clima en la NFL — protocolo pre-registrado

**Fijado el 13-nov-2026, antes de ver resultados.** Investigación: no cambia picks ni probabilidades.

- Fuente: Open-Meteo (histórico horario por coordenadas, gratis para uso no comercial; revisar licencia si la página
  se hace comercial). Hora del inicio redondeada a la hora; estadio del local (coordenadas y traslados de `travel.py`).
- Techo fijo o retráctil (Falcons ≥2017, Cowboys, Lions, Texans, Colts, Raiders ≥2020, Rams y Chargers ≥2020,
  Vikings ≥2016, Saints, Cardinals): variables en 0. Partidos neutrales: 0.
- Variables: `wind` (km/h a 10 m), `rain` (mm/h), `cold` = max(0, 5 − °C).
- **Principal — total de puntos:** residuo (total real − línea de cierre de las casas) ~ clima, regresión lineal
  walk-forward por temporada (test ≥ 2019). Éxito: reducción del error cuadrático frente a la línea sola con IC95% > 0
  (bootstrap por fecha). Si pasa, el clima ajustará la probabilidad de los picks de "Total de puntos" de NFL.
- **Secundaria — ganador:** logit(probabilidad de las casas) + clima, walk-forward; éxito si IC95% de mejora > 0.
