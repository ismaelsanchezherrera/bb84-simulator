# Simulador BB84 - Distribución de Claves Cuánticas

Simulador en Python del protocolo de distribución de claves cuánticas **BB84**. Permite analizar la estimación de la tasa de error de bits (QBER), la reconciliación de información (algoritmo Cascade) y la amplificación de la privacidad frente a ataques de intromisión (Eve).

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
