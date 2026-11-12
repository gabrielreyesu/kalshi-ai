# Kalshi AI — política de referencias (sin registrar picks a mano)

Actualizado: 2026-11-12.

## Preferencia actual
El usuario no tiene tiempo para transcribir pronósticos de X, Discord o imágenes de apuestas. Se mantienen únicamente como ilustraciones de mercados, no como datos de entrenamiento ni demostración de win-rate.

## Fuentes automáticas y permisos
- The Odds API ya conectada mediante Actions Secrets (se consulta h2h). API-Football si está configurada y la API de Skorbot como observación de referencias. Ningún scraping de sitios ni perfiles de terceros sin permiso.
- Los datos de proveedores se consideran contexto de mercado, no picks copiables. Se conservan timestamp, origen y reglas de acceso; la predicción del modelo de producción no cambia.
- Las métricas publicadas en analysts/watchlist.csv se presentan únicamente como vínculos de investigación, no modelos entrenados con contenido de los analistas. No se presume permiso para extraer, revender o entrenar sobre selecciones.
- El workflow analysts_manual.yml ya no tiene una ejecución diaria; solo se activa al editar analysts/mis_picks.csv o al ejecutarlo voluntariamente.
- El workflow external_learning.yml genera reports/external_learning/REFERENCIAS.md con estadísticas agregadas de cobertura y directorio de enlaces. No copia contenido de terceros ni recolecta picks.

## Mercados ilustrados por screenshots
Recepciones y yardas de NFL, rebotes/puntos/triples de NBA, 1X2 y tiros a puerta de fútbol, tiros a puerta y spreads de NHL. Atención: la API actual solo consulta h2h, NO props de jugadores. Cualquier expansión debe obtener endpoints adecuados, liquidaciones, líneas y derechos de uso.

## Próximas pruebas
Evaluar cobertura de las señales automáticas, estabilidad del archivo privado, matching de partidos, datos preinicio y puntuación prospectiva frente al baseline de mercado. Sin activación automática de apuestas ni promociones de candidatos.