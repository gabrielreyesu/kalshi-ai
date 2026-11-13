# ¿Acertamos mejor que Kalshi? Comparación con precios reales antes del partido

**Protocolo fijado antes de ver resultados (13-nov-2026).** Investigación: no cambia picks ni probabilidades.

## Datos
- Mercados de ganador de Kalshi **ya liquidados** (`KXNBAGAME`, `KXNFLGAME`) desde `/historical/markets` y `/markets?status=settled`.
- Velas horarias de cada mercado (`candlesticks`): se toma la última vela que **cierra al menos 60 min antes** del inicio
  (también 6 h y 24 h). Precio = punto medio bid/ask; si el spread supera 10¢, último precio negociado.
- Emparejamiento con ESPN por abreviaturas y fecha (±30 h, único o se descarta), como en producción.
- Resultado: marcador final de ESPN (gana el local o no).
- Modelo: predicciones **fuera de muestra** (walk-forward por temporada) del modelo actual y del modelo + alineación
  (NBA variante D, NFL variante B), las mismas del backtest de `LINEUP_RESEARCH.md`.

## Hipótesis
- **H1**: modelo + alineación vs Kalshi T-60 (log loss pareado, IC95% por fecha). Se espera que Kalshi gane.
- **H2**: mezcla de Kalshi T-60 y el modelo, entrenada solo con fechas anteriores (ventana creciente; 40% inicial de
  calentamiento), frente a Kalshi recalibrado igual. **Éxito: IC95% de mejora de log loss > 0.**
- Descriptivo: en partidos donde modelo y Kalshi discrepan ≥ 8 puntos, quién acierta más.

Las casas (cierre de ESPN) se muestran como referencia: su cierre es posterior a T-60, así que no es una comparación
justa en el tiempo.

## Resultados (13-nov-2026)
### NBA — 1.559 partidos (15-abr-2025 a 12-nov-2026)
| Fuente | Acierto | Log loss | Brier | ECE |
|---|---|---|---|---|
| **Kalshi T-60 min** | 68,7% | **0,5751** | 0,1971 | 0,023 |
| Kalshi T-6 h | 68,3% | 0,5774 | 0,1978 | 0,024 |
| Kalshi T-24 h | 68,4% | 0,5830 | 0,2000 | 0,018 |
| Casas (cierre ESPN) | 68,9% | 0,5755 | 0,1969 | 0,020 |
| Modelo actual | 67,7% | 0,6009 | 0,2071 | 0,018 |
| Modelo + alineación | 67,0% | 0,5910 | 0,2033 | 0,030 |

- **H1:** Kalshi T-60 gana al modelo + alineación: −0,0159 [−0,0235; −0,0085] (significativo).
- **H2:** la mezcla no mejora a Kalshi: −0,0004 [−0,0017; +0,0010] (1.018 partidos). **No supera.**
- Cuando discrepan ≥ 8 pts (350 partidos): Kalshi acierta 68,0%, el modelo 63,4%.

### NFL — 410 partidos (4-sep-2025 a 12-nov-2026)
| Fuente | Acierto | Log loss | Brier | ECE |
|---|---|---|---|---|
| **Kalshi T-60 min** | 65,6% | **0,6115** | 0,2124 | 0,030 |
| Casas (cierre ESPN) | 64,9% | 0,6121 | 0,2126 | 0,041 |
| Modelo actual | 65,4% | 0,6235 | 0,2176 | 0,062 |
| Modelo + QB titular | 65,6% | 0,6135 | 0,2129 | 0,033 |

- **H1:** −0,0020 [−0,0110; +0,0071]: prácticamente empatado con Kalshi.
- **H2:** +0,0027 [−0,0006; +0,0058] (245 partidos). **No supera**; dirección positiva, muestra pequeña. Repetir al final de temporada.
- Cuando discrepan ≥ 8 pts (98 partidos): Kalshi 61,2%, modelo 56,1%.

### Conclusiones
1. **Para acertar, la mejor probabilidad disponible es el mercado justo antes del partido** (Kalshi T-60 ≈ cierre de
   las casas). El modelo propio, incluso con alineaciones, no lo supera ni le añade información en NBA; en NFL se
   acerca pero no está demostrado.
2. **El precio mejora al acercarse el partido** (NBA: 0,5830 a T-24 h → 0,5751 a T-60): la información de última hora
   (bajas, alineaciones) importa, y el mercado la incorpora. Los picks deben usar el precio más reciente posible.
3. Las variables de alineación siguen siendo útiles donde **no hay mercado** y como respaldo; las pruebas en sombra
   continúan.
4. Corrección encontrada: Kalshi usa `LA` para los Rams; el código esperaba `LAR` y sus partidos nunca recibían
   precio de Kalshi en producción. Corregido en `kalshi_nba/kalshi.py`.

## Pick final combinado (Kalshi + casas + modelo) — en producción para NBA y NFL
Por petición explícita del usuario, el pick final de NBA/NFL pasa a ser una combinación logística de logit(Kalshi),
logit(casas) y logit(modelo) (`kalshi_nba/final_pick.py`, pesos en `kalshi_nba/final_weights.json`). Si falta una
fuente se usa el submodelo con las disponibles: (Kalshi, casas, modelo) → (Kalshi, modelo) → (casas, modelo) → modelo.

Validación fuera de muestra (pesos entrenados solo con fechas anteriores):
| | Partidos | Pick anterior (casas o modelo) | Kalshi solo | Modelo solo | Combinado |
|---|---|---|---|---|---|
| NBA | 1.018 | 71,1% · 0,5537 | 70,7% · 0,5535 | 69,4% · 0,5861 | 71,0% · 0,5554 |
| NFL | 246 | 63,8% · 0,6125 | 64,2% · 0,6131 | 65,0% · 0,6224 | 65,5% · 0,6128 |

- Con línea de las casas, el combinado **empata** con el pick anterior (NBA −0,0017 [−0,0047; +0,0013]; NFL −0,0003
  [−0,0056; +0,0046]). No es una mejora demostrada, pero tampoco empeora.
- **Sin línea de las casas** (frecuente a más de 1–2 días), el pick anterior caía al modelo solo; el combinado usa
  Kalshi + modelo, y Kalshi supera al modelo por 0,016 (NBA) en log loss.
- Pesos aprendidos: NBA Kalshi 0,55 · casas 0,55 · modelo −0,04 (el modelo no aporta frente al mercado);
  NFL Kalshi 0,47 · casas 0,36 · modelo 0,27.
- Cautela: en el histórico, "casas" es la línea de cierre (posterior a T-60); en vivo es la foto del momento.
  Kalshi solo se usa si su spread es ≤ 10¢. Reentrenar los pesos cada temporada con `python -m kalshi_nba.final_pick fit`.

## Fútbol: pick final combinado (protocolo pre-registrado, 13-nov-2026, antes de ver resultados)
- Datos: mercados liquidados de Kalshi (`KXEPLGAME`, `KXLALIGAGAME`, `KXSERIEAGAME`, `KXBUNDESLIGAGAME`, `KXLIGUE1GAME`,
  `KXUCLGAME`, `KXUELGAME`, `KXUECLGAME`, `KXUEFANLGAME`, `KXFIFAGAME`, `KXWCGAME`, `KXINTLFRIENDLYGAME`), emparejados con
  ESPN con la regla de producción (ambos nombres + hora ±36 h, único). Precio local/empate/visitante 60 min antes,
  normalizado a 1 (sin margen). Casas: mediana de cuotas ESPN sin margen. Modelo: `kalshi_intl` fuera de muestra.
- Combinador: agregación log-lineal con pesos por fuente (`combine3`), submodelos si falta una fuente.
- Validación: ventana creciente por fecha (40% inicial de calentamiento). Se compara con el pick actual (casas si hay;
  si no, modelo) y con Kalshi. **Criterio para producción** (pedido explícito del dueño de usar todas las fuentes): que
  el combinado **no empeore** al pick actual (límite superior del IC95% de la diferencia de log loss ≥ 0, es decir,
  que no sea significativamente peor). Se reporta por grupo: ligas, copas y selecciones.

**Aclaración antes del resultado final:** "pick actual" significa el pick que hoy usa la página en cada grupo. En las
ligas la página no tiene línea de las casas en vivo (usa el modelo), así que en ligas se compara Kalshi + modelo frente
al modelo, y el combinador se entrena sin casas para ligas. En copas y selecciones: frente a las casas.

### Resultado fútbol (13-nov-2026, 1.510 partidos emparejados; ligas aún descargando) — SUPERA el criterio
Validación fuera de muestra (905 partidos evaluables, 17-feb a 13-nov-2026):
| Grupo | Partidos | Pick actual de la página | Combinado | Mejora log loss [IC95%] |
|---|---|---|---|---|
| Total | 905 | 58,5% · 0,9134 | 59,0% · 0,9028 | +0,0106 [−0,0030; +0,0253] |
| Ligas (vs modelo) | 219 | 1,0361 | 1,0262 | +0,0099 [−0,0181; +0,0387] |
| Copas (vs casas) | 256 | 0,9124 | 0,9164 | −0,0040 [−0,0210; +0,0146] |
| Selecciones (vs casas) | 430 | 0,8515 | 0,8318 | +0,0196 [−0,0031; +0,0452] |

Criterio pre-registrado (no ser significativamente peor que el pick actual): se cumple en total y en cada grupo.
Pesos (Kalshi + casas + modelo): Kalshi 0,56 · casas 0,49 · modelo 0,15. Reentrenar con todas las ligas al terminar
la descarga (`python -m kalshi_lineup.soccer_kalshi_bench`).
