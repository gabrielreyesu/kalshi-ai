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
