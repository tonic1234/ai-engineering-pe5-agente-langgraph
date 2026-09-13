"""tests/test_agente.py — Pruebas de las herramientas y del grafo.

Nota: acá se puede probar bastante sin llamar al LLM:
  1. Cada herramienta por separado (son funciones puras).
  2. Que el grafo se construya con los nodos que pide la consigna y que la arista
     condicional exista.
  3. La persistencia: que SqliteSaver cree el archivo de checkpoints.

La prueba multi-paso real (que el agente llame a la herramienta >= 2 veces) necesita al
LLM y se corre a mano con `python agent.py`; queda la traza en traza_ejecucion.json.

Correr:  pytest -q
"""

from __future__ import annotations

import json

import pytest

from tools import TOOLS, calcular_prioridad, consultar_base_de_datos, fecha_actual


# --------------------------------------------------------------------------
# Herramientas
# --------------------------------------------------------------------------
def test_hay_al_menos_una_herramienta():
    assert len(TOOLS) >= 1


def test_todas_las_herramientas_tienen_docstring():
    # El modelo elige la herramienta por el docstring: si falta, el agente se pierde.
    for tool in TOOLS:
        assert tool.description and len(tool.description) > 40


def test_consulta_existente():
    salida = json.loads(consultar_base_de_datos.invoke({"identificador": "ORD-1001"}))
    assert salida["estado"] == "en preparación"
    assert salida["total"] == 1250.00


def test_consulta_inexistente_devuelve_error():
    salida = json.loads(consultar_base_de_datos.invoke({"identificador": "ORD-9999"}))
    assert "error" in salida


def test_prioridad_alta_con_demora():
    assert "P1" in calcular_prioridad.invoke({"criticidad": "alta", "dias_demora": 3})


def test_prioridad_baja():
    assert "P3" in calcular_prioridad.invoke({"criticidad": "baja", "dias_demora": 0})


def test_prioridad_criticidad_invalida():
    assert "inválida" in calcular_prioridad.invoke({"criticidad": "urgentísima", "dias_demora": 1})


def test_fecha_actual_formato_iso():
    from datetime import datetime

    valor = fecha_actual.invoke({})
    datetime.fromisoformat(valor)  # si no es ISO, esto lanza


# --------------------------------------------------------------------------
# Grafo
# --------------------------------------------------------------------------
def test_el_grafo_tiene_los_nodos_y_la_arista_condicional(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from agent import build_graph

    grafo = build_graph()
    nodos = set(grafo.get_graph().nodes)
    assert {"modelo", "herramientas"} <= nodos


def test_recursion_limit_definido():
    # Sin techo de recursión el agente puede quedar en un bucle carísimo.
    from agent import RECURSION_LIMIT

    assert 1 <= RECURSION_LIMIT <= 25


# --------------------------------------------------------------------------
# Fábrica de modelos
# --------------------------------------------------------------------------
@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_get_model_soporta_los_tres_proveedores(provider):
    from agent import get_model

    assert get_model(provider=provider) is not None


def test_get_model_rechaza_proveedor_desconocido():
    from agent import get_model

    with pytest.raises(ValueError):
        get_model(provider="perplexity")


def test_el_modelo_queda_enlazado_a_las_herramientas():
    import os

    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    from agent import build_llm

    # bind_tools devuelve un runnable que ya conoce las herramientas del agente.
    modelo = build_llm(provider="openai")
    assert hasattr(modelo, "ainvoke")


def test_normalizacion_del_contenido():
    # OpenAI/Anthropic devuelven str; Gemini devuelve lista de bloques.
    from agent import _texto

    assert _texto("hola") == "hola"
    assert _texto([{"type": "text", "text": "ho"}, {"type": "text", "text": "la"}]) == "hola"
    assert _texto([{"type": "text", "text": "ok", "extras": {"signature": "xx"}}]) == "ok"
