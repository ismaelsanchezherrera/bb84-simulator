# Simulador BB84 - Distribución de Claves Cuánticas

Simulador en Python del protocolo de distribución de claves cuánticas **BB84**. Permite analizar la estimación de la tasa de error de bits (QBER), la reconciliación de información y la amplificación de la privacidad frente a ataques de intromisión.

## Estructura del Proyecto

- `src/bb84_simulator/`: Código fuente del simulador.
- `tests/`: Batería de pruebas unitarias y de integración.

## Instalación en Desarrollo

```bash
git clone [https://github.com/ismaelsanchezherrera/bb84-simulator.git](https://github.com/ismaelsanchezherrera/bb84-simulator.git)
cd bb84-simulator
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```
