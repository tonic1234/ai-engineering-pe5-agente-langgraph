"""tools.py — Herramientas del agente.

APUNTE (lo que más me costó entender):
El agente NO tiene rutas fijas. El LLM lee el DOCSTRING de cada herramienta y decide
solo cuál usar según lo que le pidió el usuario. Por eso el profe insistió: si el agente
no usa la herramienta que esperabas, el problema está en la descripción, no en el grafo.

Por eso escribí los docstrings bien explícitos, diciendo qué devuelve cada una y en qué
caso conviene usarla.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

# Base "de juguete" para simular una consulta a una base de datos técnica.
DB_PATH = Path("./fichas.json")

_FICHAS_DEFAULT = {
    "ORD-1001": {"estado": "en preparación", "cliente": "Acme", "total": 1250.00},
    "ORD-1002": {"estado": "despachada", "cliente": "Globex", "total": 480.50},
    "ORD-1003": {"estado": "cancelada", "cliente": "Initech", "total": 0.0},
    "CLI-ACME": {"plan": "enterprise", "contrato_hasta": "2027-03-01"},
}


@tool
def consultar_base_de_datos(identificador: str) -> str:
    """Busca una orden o un cliente en la base de datos técnica interna.

    Usá esta herramienta cuando necesites datos concretos de una orden (estado, cliente,
    total) o de un cliente (plan, vigencia del contrato). El identificador tiene el
    formato 'ORD-####' para órdenes o 'CLI-XXX' para clientes.

    Args:
        identificador: código de la orden (ej. ORD-1001) o del cliente (ej. CLI-ACME).

    Returns:
        Un JSON con los datos encontrados, o un mensaje de error si el identificador
        no existe. Si el identificador no existe, el agente debería pedirle al usuario
        que lo verifique en vez de inventar datos.
    """

    db = json.loads(DB_PATH.read_text(encoding="utf-8")) if DB_PATH.exists() else _FICHAS_DEFAULT
    registro = db.get(identificador.upper())

    if registro is None:
        return json.dumps({"error": f"No existe el identificador {identificador!r}. Verificalo."})

    return json.dumps({"identificador": identificador.upper(), **registro})


@tool
def calcular_prioridad(criticidad: str, dias_demora: int) -> str:
    """Calcula la prioridad de un incidente a partir de su criticidad y los días de demora.

    Usá esta herramienta cuando tengas la criticidad (alta, media o baja) y cuántos días
    lleva el incidente sin resolverse. Devuelve la prioridad final y una recomendación.

    Args:
        criticidad: 'alta', 'media' o 'baja'.
        dias_demora: cantidad de días que el incidente lleva abierto (entero >= 0).

    Returns:
        Texto con la prioridad calculada (P1, P2 o P3) y el SLA recomendado.
    """

    criticidad = criticidad.strip().lower()
    if criticidad not in {"alta", "media", "baja"}:
        return f"Criticidad inválida: {criticidad!r}. Usá alta, media o baja."

    if criticidad == "alta" and dias_demora >= 2:
        return "Prioridad P1 — escalar ahora, SLA 4 horas."
    if criticidad == "alta":
        return "Prioridad P2 — SLA 24 horas."
    if criticidad == "media":
        return "Prioridad P2 — SLA 48 horas." if dias_demora > 5 else "Prioridad P3 — SLA 5 días."
    return "Prioridad P3 — SLA 10 días."


@tool
def fecha_actual() -> str:
    """Devuelve la fecha y hora actual del sistema en formato ISO (UTC).

    Usala cuando el usuario pregunte 'hoy', 'ahora' o necesites calcular una diferencia
    de días y no tengas la fecha de referencia.
    """

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


TOOLS = [consultar_base_de_datos, calcular_prioridad, fecha_actual]
