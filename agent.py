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
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOLS

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Límite de pasos: sin esto el agente puede entrar en un bucle y gastar tokens sin parar.
RECURSION_LIMIT = 10
CHECKPOINT_DB = Path("./checkpoints.sqlite")


def build_llm(model: str = "gpt-4o-mini"):
    """Modelo con las herramientas "enlazadas" (bind_tools)."""

    llm = ChatOpenAI(model=model, temperature=0)
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
            if getattr(mensaje, "content", None):
                entrada["contenido"] = mensaje.content
            if getattr(mensaje, "tool_calls", None):
                entrada["herramientas"] = [
                    {"nombre": c["name"], "argumentos": c["args"]} for c in mensaje.tool_calls
                ]
            traza.append(entrada)

        return {"respuesta": resultado["messages"][-1].content, "traza": traza}


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
