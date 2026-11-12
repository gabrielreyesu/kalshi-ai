# Kalshi AI V8 — Aprendizaje de predicciones externas (investigación)

## Objetivo
Añadir señales externas al motor sin dejar que fuentes no validadas reescriban predicciones propias. El sistema consulta proveedores oficiales, guarda **snapshots anteriores al inicio**, y estima calidad de fuentes de forma emparejada frente a nuestro modelo. No copia automáticamente selecciones de apuestas.

### Fuentes soportadas

| Fuente | Tipo | Estado sin credenciales |
| --- | --- | --- |
| The Odds API | Consenso de cuotas desmarjinadas por partido (NO es un analista individual) | Saltada |
| API-Football | Pronósticos estadísticos 1X2 de otro proveedor | Saltada |
| Skorbot (skorbot.live) | API pública de predicciones de fútbol; observación y evaluación con atribución (sin entrenamiento a partir de sus datos) | Activada sin clave |
| Feed JSON autorizado de analistas | Pronósticos 1X2 o 2 vías con autor, fecha de publicación y permiso explícito | Saltado |

No scrapeamos webs ni perfiles de personas. Metaculus y Football-Data no se integran porque sus condiciones actuales restringen este uso sin autorización. Respeta todas las cuotas, privacidad y licencias de proveedores.

## Qué hace
- Captura pronósticos antes de cada inicio y exige coincidencia **única** de equipo, competición y hora con un `event_id` de nuestro registro, sin heurísticas arriesgadas.
- Conserva `source_id`, `published_utc`, `captured_utc`, `kickoff_utc`, probabilidades y las probabilidades originales de nuestro modelo.
- Elimina el margen por cada casa en el consenso y promedia; exige al menos 2 casas con cuotas recientes.
- Mantiene el historial de señales en `data/external_signals/` (ignorado en git, restaurado mediante cache GitHub Actions).
- Todos los días a las **01:35 y 13:35 UTC** ejecuta una recogida; los lunes el flujo histórico compara las señales registradas y los resultados de internacionales concluidos al tiempo reglamentario.
- Produce `reports/external_learning/source_performance.csv` y `status.json`, pero **no publica históricos brutos ni claves**.
- Requiere por lo menos 30 partidos puntuados por fuente para marcarla *elegible para revisión*; **NO activa pesos ni modifica producción**. Comparar contra mercados debe utilizar precios/observaciones capturados antes del partido.

## Cómo habilitar proveedores (no pegar las claves aquí ni subirlas al repositorio)
En GitHub → Settings → Secrets and variables → Actions → New repository secret:
- `THE_ODDS_API_KEY`: clave de The Odds API (opcional).
- `API_FOOTBALL_KEY`: clave de API-Football (opcional).
- `LICENSED_ANALYST_FEED_URL` y `LICENSED_ANALYST_FEED_HOST`: HTTPS exacto y host permitido de un feed con autorización explícita para análisis y modelado (opcional).

Si no están configuradas, la automatización sigue ejecutándose, pero registra **fuentes saltadas**; no inventa datos ni simula que está aprendiendo de otros analistas. Recomendación: no comprometer a un repositorio público datos que los términos de la fuente prohíban redistribuir.

### Formato de analista autorizado
El endpoint HTTPS debe devolver un arreglo JSON:
```json
[
  {
    "licensed_for_modeling": true,
    "sport": "intl",
    "event_id": "ESPN_EVENT_ID",
    "source_id": "analyst_contract_id",
    "published_utc": "2026-11-13T12:00:00Z",
    "p_H": 0.54,
    "p_D": 0.23,
    "p_A": 0.23
  }
]
```
Para NFL y NBA, `sport` es `nfl` o `nba` y no se usa `p_D`. La fecha de publicación debe ser verificable y previa al inicio. No se extraen automáticamente probabilidades de texto libre.

## Límites
- Los registros de ESPN y de API-Football pueden tener diferencias de nombre o no estar cubiertos; cuando no hay coincidencia inequívoca se ignora.
- Para fuentes externas, solo se calculan métricas de resultados internacionales con partidos que hayan entrado en nuestro backtest OOS y estén finalizados a 90 minutos. NBA/NFL se recolectan, pero su asentamiento de resultados requiere otro validador.
- El cache de GitHub Actions **no es almacenamiento permanente**. Para aprendizaje a largo plazo y proveedores con restricciones, habrá que configurar almacenamiento privado durable. Una caída o expiración del cache puede perder la historia.
- La V8 no puede demostrar todavía ninguna mejora de accuracy, Brier o log loss atribuible a terceros hasta acumular suficientes pronósticos reales prospectivos.
- La ejecución no es continua literalmente: es programada, depende de disponibilidad de GitHub Actions, proveedores y cuota.
- El workflow semanal solo vuelve a ajustar el candidato propio; el mezclador de pronósticos externos sigue en investigación hasta validación independiente.

### Fuente abierta Skorbot
El sistema consulta su API JSON pública de forma moderada con `User-Agent` identificable y atribución **Skorbot (skorbot.live)**. Descarta partidos ya finalizados, resultados presentes, predicciones posteriores al saque inicial y cualquier encuentro sin identificación inequívoca. Su calendario publicado puede estar desactualizado; la ausencia de nuevos pronósticos no se interpreta como error del modelo. En tanto no esté claro el permiso de entrenamiento, se limita al monitoreo y no se utiliza como señal para ajustar nuestros coeficientes.

### Aprendizaje experimental de fuentes autorizadas
El proceso semanal también ajusta pesos candidatos de mezcla **solo con partidos anteriores** de fuentes autorizadas. Selecciona entre pesos 0%, 10%, 20%, 30% y 40% usando el 70% cronológicamente más antiguo y evalúa el 30% posterior. Requiere mínimo 60 partidos históricos de entrenamiento y 25 de prueba, sin mezclar partidos del mismo día entre entrenamiento y evaluación. Emite `candidate_weights.json`. Es investigación: **ningún peso se aplica a producción y una mejora exploratoria no garantiza rendimiento futuro**. Las fuentes con permiso de entrenamiento no confirmado, incluido Skorbot, quedan excluidas.
