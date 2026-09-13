"""agent.py — Grafo ReAct con LangGraph y memoria persistente.

APUNTE DE CLASE (el corazón de la pre-entrega):
El agente es un grafo de dos nodos que se repiten:
    modelo -> (¿pidió una herramienta?) -> herramientas -> modelo -> ... -> fin

Esa repetición es el "ciclo ReAct". La arista que decide si sigue el ciclo o termina es
una arista CONDICIONAL: tools_condition mira si el último mensaje del modelo trae
tool_calls; si trae, va al nodo de herramientas, si no, termina.

El estado hereda de MessagesState. Ojo: el estado se ACUMULA (los mensajes se van
sumando), no se reemplaza. Eso lo maneja el reducer que trae MessagesState por defecto.

La persistencia la da el checkpointer: guarda el estado por thread_id, así el agente
recuerda lo que hablaron antes en la misma sesión. Sin eso, el agente es efímero.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOLS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Modelo por defecto de cada proveedor.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-flash-latest",  # free tier, sin tarjeta
}

# Límite de pasos: sin esto el agente puede entrar en un bucle y gastar tokens sin parar.
RECURSION_LIMIT = 10
CHECKPOINT_DB = Path("./checkpoints.sqlite")


def get_model(provider: str | None = None, model: str | None = None, temperature: float = 0.0):
    """Fábrica de modelos: devuelve el modelo del proveedor pedido.

    Misma idea que en los módulos anteriores: el proveedor se elige por variable de
    entorno (LLM_PROVIDER), así cambiar de modelo no toca la lógica del grafo.
    """

    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise ValueError(f"Proveedor no soportado: {provider!r} (opciones: openai, anthropic, gemini)")


def build_llm(model: str | None = None, provider: str | None = None):
    """Modelo con las herramientas "enlazadas" (bind_tools)."""

    llm = get_model(provider=provider, model=model, temperature=0)
    return llm.bind_tools(TOOLS)


def build_graph(checkpointer=None):
    """Arma el StateGraph: nodo modelo + nodo herramientas + arista condicional."""

    llm_with_tools = build_llm()

    async def call_model(state: MessagesState):
        respuesta = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [respuesta]}

    builder = StateGraph(MessagesState)
    builder.add_node("modelo", call_model)
    builder.add_node("herramientas", ToolNode(TOOLS))  # ejecuta las tool_calls

    builder.set_entry_point("modelo")
    # tools_condition: si el modelo pidió herramientas -> "herramientas"; si no -> END.
    # Ojo: tools_condition devuelve el literal "tools", así que tengo que mapearlo al
    # nombre que le puse a mi nodo (si no, LangGraph se queja del destino desconocido).
    builder.add_conditional_edges(
        "modelo",
        tools_condition,
        {"tools": "herramientas", END: END},
    )
    builder.add_edge("herramientas", "modelo")  # el ciclo vuelve al modelo

    return builder.compile(checkpointer=checkpointer)


def _texto(content) -> str:
    """Normaliza el contenido de un mensaje a texto plano.

    Ojo: OpenAI y Anthropic devuelven un string, pero Gemini devuelve una LISTA de
    bloques. Sin esto, la respuesta final sale como una lista de diccionarios.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = []
        for bloque in content:
            if isinstance(bloque, str):
                partes.append(bloque)
            elif isinstance(bloque, dict) and bloque.get("type") == "text":
                partes.append(bloque.get("text", ""))
        return "".join(partes)
    return str(content)


async def run_agent(consulta: str, thread_id: str = "sesion-1", model: str = "gpt-4o-mini") -> dict:
    """Ejecuta el agente y devuelve la traza completa (para el .json de la entrega)."""

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

        resultado = await graph.ainvoke(
            {"messages": [HumanMessage(content=consulta)]},
            config=config,
        )

        traza = []
        for mensaje in resultado["messages"]:
            entrada = {"tipo": mensaje.__class__.__name__}
            texto = _texto(getattr(mensaje, "content", "") or "")
            if texto:
                entrada["contenido"] = texto
            if getattr(mensaje, "tool_calls", None):
                entrada["herramientas"] = [
                    {"nombre": c["name"], "argumentos": c["args"]} for c in mensaje.tool_calls
                ]
            traza.append(entrada)

        return {"respuesta": _texto(resultado["messages"][-1].content), "traza": traza}


if __name__ == "__main__":
    consulta = os.getenv(
        "CONSULTA",
        "Necesito el estado de la orden ORD-1001 y, según su criticidad alta con 3 días "
        "de demora, qué prioridad le corresponde. Dame también la fecha de hoy.",
    )

    salida = asyncio.run(run_agent(consulta))
    print("\n=== RESPUESTA ===")
    print(salida["respuesta"])

    Path("traza_ejecucion.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTraza guardada en traza_ejecucion.json")
