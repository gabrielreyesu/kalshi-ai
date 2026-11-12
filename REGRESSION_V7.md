# V7: comparación emparejada obligatoria

GitHub Actions ejecutará el modelo V6 (base main) y el modelo V7 propuesto en el mismo conjunto de datos descargado de ESPN y guardará `reports/intl_regression_comparison/`.

El PR NO se fusionará automáticamente. La comparación se basa en accuracy, log loss, Brier e intervalos de confianza por fecha. Si un motor evalúa un conjunto distinto de partidos, la herramienta fallará en vez de comparar muestras incompatibles.

La suite local alcanzó 61 pruebas aprobadas y 2 advertencias de pandas. Se requiere completar esta ejecución real antes de desplegar los cambios en producción.