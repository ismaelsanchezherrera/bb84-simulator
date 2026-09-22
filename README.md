# Simulador BB84 - Distribución de Claves Cuánticas

Simulador en Python del protocolo de distribución de claves cuánticas **BB84**. Permite analizar la estimación de la tasa de error de bits (QBER), la reconciliación de información (algoritmo Cascade) y la amplificación de la privacidad frente a ataques de intromisión (Eve).

> **Hipótesis del Modelo**: El protocolo asume la existencia de un canal clásico autenticado exógeno. La fase de confirmación de clave (`confirmacion_clave_ok`) comprueba la coincidencia de las claves resultantes (mediante hashes/tags), pero no autentica la identidad de las partes frente a ataques de tipo Man-In-The-Middle (MITM) en el canal clásico.

## Hipótesis Físicas y Marco de Seguridad

Este simulador implementa el protocolo BB84 bajo las siguientes hipótesis de modelo:

1. **Canal Clásico Autenticado Exógeno:** La fase de confirmación de clave (`confirmacion_clave_ok`) comprueba la igualdad de las claves tras la reconciliación. El protocolo asume que la autenticación frente a ataques Man-In-The-Middle (MITM) en el canal clásico está garantizada por un canal clásico preautenticado exógeno.
2. **Cota de Error de Fase (Serfling):** La estimación del error de fase a partir del QBER de bit observado supone un canal cuántico simétrico $e_x \approx e_z$ (hipótesis de ruido despolarizante / ataques individuales sin memoria).
3. **Seguridad Teórico-Informacional (ITS):** En modo `EntropySource.crypto()`, la matriz de Toeplitz para la amplificación de privacidad se genera mediante entropía criptográfica directa (`os.urandom`), garantizando la cota de secreto de la Leftover Hash Lemma ($2^{-\ell}$) sin depender de un PRNG de estado acotado.
4. **Modelo Físico de Detectores Independientes:** La etapa de medición en Bob simula la llegada de fotones y cuentas oscuras sobre dos detectores físicos independientes ($D_0$ y $D_1$), contemplando la resolución estocástica ante eventos de doble clic.

## Estructura del Proyecto

- `src/bb84_simulator/`: Código fuente e implementación de la simulación.
- `examples/`: Scripts demostrativos y casos de uso.
- `tests/`: Batería de pruebas unitarias y de integración.

## Requisitos Previos

- Python 3.9 o superior

## Instalación en Desarrollo

Clona el repositorio e instala el paquete en modo editable junto con sus dependencias de desarrollo:

```bash
git clone [https://github.com/ismaelsanchezherrera/bb84-simulator.git](https://github.com/ismaelsanchezherrera/bb84-simulator.git)
cd bb84-simulator
python -m venv .venv
source .venv/Scripts/activate  # En Git Bash en Windows
pip install -e ".[dev]"
```
