# Kalshi AI: fútbol, NBA, NCAA (baloncesto) y NFL

Reemplaza las ~60 celdas del notebook de fútbol por un paquete reproducible. Mismo modelo (V4 híbrido:
Elo + Poisson/Dixon-Coles + regresión logística por liga), sin variables globales de Colab, y con
**todos los mercados** salidos de la misma matriz de marcadores.

| Mercado | Columna en el registro | Validación |
|---|---|---|
| Ganador / empate (1X2) | `p_H`, `p_D`, `p_A` | backtest + comparación contra cuotas de cierre |
| Totales over 1.5 / 2.5 / 3.5 | `p_over_X_raw`, `p_over_X_cal` | backtest; over 2.5 contra cuotas de cierre |
| Ambos anotan | `p_btts_raw/cal` | backtest |
| Spread ±1.5 (gana por 2+) | `p_home_m1_5_raw/cal`, `p_away_m1_5_raw/cal` | backtest |

`raw` = Poisson directo. `cal` = recalibrado con regresión logística. Usa el que gane en `reports/soccer_backtest_report.txt`.

## En Colab (sin instalar nada raro)
```python
!unzip -q kalshi_ai.zip && cd kalshi_ai && pip -q install -r requirements.txt
%cd kalshi_ai
!python -m unittest discover -s tests          # debe decir OK
!python -m kalshi_soccer.cli backtest          # descarga datos, evalúa 2022/23 -> hoy, guarda reports/
```
Lee `reports/soccer_backtest_report.txt`: lo importante es la sección **"MISMO SUBCONJUNTO CON CUOTAS DE MERCADO"**.

## Subir a GitHub (opción fácil desde Colab)
1. En github.com crea un repositorio **vacío**.
2. Crea un token con acceso solo a ese repo (permisos Contents y Workflows: Read and write).
3. En Colab: `%run push_to_github.py` (pide usuario, repo y token; el token no se guarda).
   (Si subes archivos arrastrándolos en la web de GitHub, la carpeta oculta `.github` suele quedarse fuera.)

## Automático (GitHub Actions, gratis)
1. El repositorio ya contiene los dos flujos: `soccer-backtest` (manual) y `soccer-daily` (cada día).
2. Settings → Secrets and variables → Actions → New repository secret: `FOOTBALL_DATA_API_KEY`.
3. Pestaña Actions → `soccer-backtest` → Run workflow: el reporte aparece en el resumen de la corrida y como archivo descargable.
   Luego `soccer-daily` corre solo cada día (o lánzalo a mano).
4. Cada corrida: tests → evalúa partidos jugados → predice 14 días → guarda `predictions/registry.csv`
   (solo agrega filas; el historial de git es la auditoría).

## Reglas de seguridad ya incluidas
- Un partido solo se registra si aún no empezó y nunca se reescribe una fila existente.
- Equipos que no se pueden mapear NO se adivinan: salen en `predictions/unresolved_teams.csv`.
- Equipos nuevos en la liga (ascendidos sin partidos en el histórico) se omiten salvo `--allow-new`.
- Los tests verifican: paridad con tu notebook (diferencia < 1e-9), que cambiar resultados futuros no altera
  variables pasadas, que sin señal real el modelo no "gana" a la línea base, y el flujo predecir→registrar→evaluar.

## Límites (honestos)
- No usa alineaciones, lesiones ni xG. Solo resultados históricos.
- Las cuotas históricas vienen de football-data.co.uk; los nombres de columnas se detectan automáticamente,
  pero hay que confirmar en la primera corrida real qué fuente usó (el reporte lo imprime).
- Aún NO compara contra precios de Kalshi (siguiente capa).

## Resultados del backtest real (2022/23 -> hoy, 7,582 partidos, fuera de muestra)
- 1X2: modelo logloss 0.9871 vs cuotas de cierre 0.9651 (el mercado gana, IC95% [0.018, 0.026]).
- Over 2.5: modelo calibrado 0.6815 vs cuotas 0.6667.
- Mezcla modelo + cuotas (`python -m kalshi_soccer.cli blend`): no mejora a las cuotas solas (IC incluye 0).
  Conclusión: con solo resultados históricos el modelo no aporta información sobre el mercado.

## Precios de Kalshi
`predict` ahora une el precio de Kalshi (bid/ask de local/empate/visitante, API pública) a cada partido y guarda:
- `predictions/registry.csv`: predicción + precios al momento del registro (columnas `k_*`).
- `predictions/kalshi_snapshots.csv`: foto de precios en cada corrida diaria.
- `predictions/soccer_kalshi_edges.csv`: ventaja BRUTA (p_modelo - ask), sin comisiones. Verifica la comisión vigente de Kalshi.
Un nombre de equipo que no coincide de forma única deja el partido sin precio (tabla `KALSHI_ALIASES` en `kalshi.py`).

---
# NBA (`kalshi_nba`)
Datos: API pública de ESPN (resultados desde 2015/16 y líneas históricas de varias casas). La línea de mercado es la
**mediana entre casas** (una casa traía totales corruptos en 2020-2023; la mediana lo neutraliza).
Motor: Elo con multiplicador por margen, regresión entre temporadas, descanso/back-to-back, forma y anotación reciente.
Mercados: ganador, spread y total (normal alrededor del margen/total esperado, sigma de entrenamiento).

```
python -m kalshi_nba.cli backtest    # ~5 min la primera vez (descarga cuotas de 14 mil partidos)
python -m kalshi_nba.cli predict     # registro + foto de precios de Kalshi (KXNBAGAME / KXNBASPREAD / KXNBATOTAL)
python -m kalshi_nba.cli evaluate    # modelo vs precio de Kalshi en mercados resueltos
```
Resultado del backtest real (10,216 partidos fuera de muestra, 2018/19 -> hoy):
- Ganador: modelo 65.1% / logloss 0.6216; mercado 67.6% / 0.5984 (el mercado gana, IC95% [0.019, 0.027]).
- Margen RMSE: modelo 13.87 vs línea 13.42. Total MAE: modelo 14.98 vs línea 14.32.
- Over/under y cobertura del spread contra la línea: 50.0% y 49.4% de acierto (sin ventaja).
- Mezcla línea + modelo: no mejora a la línea sola.
`predictions/nba_kalshi_edges.csv` muestra la ventaja BRUTA (sin comisiones). Las diferencias grandes suelen ser
lesiones o descansos que el modelo no ve, no oportunidades.

---
# NCAA baloncesto y NFL (mismo motor, `--league`)
```
python -m kalshi_nba.cli backtest --league ncaab   # ~7 min la 1ª vez (48 mil partidos)
python -m kalshi_nba.cli backtest --league nfl
python -m kalshi_nba.cli predict  --league nfl      # registro + precios Kalshi (KXNFLGAME / KXNFLSPREAD / KXNFLTOTAL)
python -m kalshi_nba.cli evaluate --league nfl
```
| Liga | Partidos fuera de muestra | Acierto modelo | Logloss modelo | Logloss mercado | ¿Mezcla mejora a la línea? |
|---|---|---|---|---|---|
| NBA | 10,216 | 65.1% | 0.6216 | 0.5984 | No |
| NFL | 2,096 | 64.3% | 0.6352 | 0.6093 | No |
| NCAA | 28,989 (con línea) | 71.2% | 0.5555 | 0.5193 | No |

Pendiente: `predict` para NCAA (unir ~360 equipos de ESPN con los nombres de Kalshi). El flujo `nba-nfl-daily` registra
NBA y NFL cada día con la foto de precios de Kalshi; `evaluate` compara modelo vs Kalshi cuando los mercados se resuelven.
