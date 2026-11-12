# V6: torneo experimental de calibración temporal

Se comparan, por separado y sobre idénticos partidos OOS, el modelo original, regresión logística multinomial, mezcla 25%, mezcla 50% y calibración por temperatura. Cada temporada usa solo resultados de temporadas anteriores.

Resultados exploratorios del histórico disponible (3,101 partidos evaluados):

| Variante | Accuracy | Log loss | Brier |
|---|---:|---:|---:|
| Original | 58.30% | 0.918130 | 0.539588 |
| Logística completa | 58.08% | 0.916851 | 0.540108 |
| Mezcla 25% | 58.40% | 0.915891 | 0.539043 |
| Mezcla 50% | 58.30% | 0.915166 | 0.538948 |
| Temperatura | 58.30% | 0.914689 | 0.538936 |

**Advertencia:** estas variantes se inspeccionaron y eligieron durante la investigación del mismo conjunto histórico, incluido 2026. Las estimaciones y sus intervalos no equivalen a confirmación de superioridad futura ni se han corregido por selección múltiple. En 2026, algunas variantes empeoran el log loss frente al original.

El workflow guardará `reports/intl_experiments/` cada lunes, para observar el rendimiento nuevo sin sustituir automáticamente el modelo de producción. El archivo JSON de entrenamiento V5 se mantiene como candidato. No se realizan apuestas ni cambios en el motor en producción.
