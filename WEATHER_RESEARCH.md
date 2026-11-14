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

## Resultado (14-nov-2026) — NO_SUPERA_CRITERIO (totales no concluyente; ganador empeora)
3.164 partidos; 2.190 al aire libre (clima de Open-Meteo a la hora del inicio, descargado día a día).
- Totales frente a la línea sola: +1,67 puntos² [+0,01; +3,37]. **Pero** la regresión incluye un intercepto que corrige
  el sesgo medio de la línea (+0,43 puntos); frente a un ajuste que solo corrige ese sesgo, el clima aporta
  +1,47 [−0,13; +3,14] (todos) y +1,12 [−0,91; +3,19] (aire libre): **no significativo**.
- Viento ≥ 25 km/h (101 partidos): residuo real −0,39 puntos frente a −2,37 que ajustaría el modelo: la línea de las
  casas ya incorpora el viento.
- Coeficientes (puntos por unidad): viento −0,11 por km/h, lluvia −1,76 por mm/h, frío −0,07 por °C bajo 5 °C.
- Ganador (mercado + clima): −0,0014 [−0,0026; −0,0002]: empeora.
Decisión: no se incorpora. Repetir con más temporadas; probar frente a la línea de APERTURA (antes de que el mercado
reaccione al pronóstico), no solo la de cierre.
