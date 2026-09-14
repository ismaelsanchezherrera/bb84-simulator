# Changelog

Todos los cambios notables en este proyecto se documentarán en este archivo.

## [5.4.1] - 2026-09-14

### Correcciones de Robustez y Tipado

- **Tratamiento de `NaN` e `inf`:** `_validar_no_negativo()` y `fec_efficiency` ahora exigen `math.isfinite()`. Evita que valores no numéricos evadan las comparaciones de una sola cota.
- **Validación de enteros en Cascade:** `error_correction_cascade` valida explícitamente que `n_pasadas` sea un número entero (excluyendo `bool`), evitando un `TypeError` interno si se ingresa un decimal.

## [5.4.0]- 2026-09-01

### Correcciones de Seguridad

- **Validación estricta de parámetros:** `SecurityParameters` ahora exige que todos los `epsilon` estén en el intervalo abierto (0, 1), `fec_efficiency >= 1` y `tag_length` sea entero positivo o `None`.
- **Validación en canales:** `FiberChannel`, `DepolarizingChannel` y `FreeSpaceChannel` validan no-negatividad de distancias/atenuaciones y rangos [0, 1] de probabilidades/eficiencias.
- **Capa Clásica:** `parameter_estimation` exige `0 < fraccion_verificacion < 1`. `error_correction_cascade` valida `qber_estimado` en [0, 0.5] y rechaza longitudes de clave distintas.
- **Fuente de entropía:** `EntropySource.choice()` soporta `size` como tupla y `a` entero en modo `crypto`.

### Cambios de Arquitectura

- **Inyección de canal:** `BB84Simulator.run()` acepta el parámetro opcional `channel: ChannelModel` para inyectar canales como `DepolarizingChannel` o `FreeSpaceChannel` directamente.
- **Reportes:** `SecurityReport.distancia_km` pasa a ser opcional (`None` cuando el canal no usa fibra). Se reemplaza `expected_qber` por la propiedad de solo lectura `qber_medido`.

### Pruebas

- **Prueba de independencia:** Se añadió una tabla de contingencia 2x2 con test de chi-cuadrado para comprobar la independencia de bits y bases en `run_statistical_tests()`.

## [5.3.0]- 2026-08-015

### Correcciones Formales

- **Ajuste de Serfling:** Se aplicó el factor de corrección $N/(N-n)$ sobre el margen de Serfling para acotar el error sobre los bits restantes no muestreados (acorde a Fung et al. 2010 / Tomamichel et al. 2012).
- **Control de clave vacía:** `evaluate_and_build` marca aborto explícito cuando la amplificación de privacidad devuelve 0 bits.

### Rendimiento

- **Optimización en Cascade:** Actualización $O(1)$ de paridades mediante XOR al corregir bits, cálculo de paridades iniciales con `np.add.reduceat` y mapeo de bloques vectorizado.

## [5.2.0]- 2026-08-01

### Arquitectura Principal

- Implementación de la arquitectura en 3 capas (Quantum, Classical, Security).
- Soporte para Leftover Hash Lemma (LHL), confirmación de clave por familias de hash universal y clases abstractas para canales y estrategias de espionaje (`EveStrategy`).
