# Agente de razonamiento cíclico con memoria persistente

Pre-entrega 5 del curso **AI Engineering** (Coderhouse).
Agente **ReAct** construido con **LangGraph**: el modelo decide solo qué herramienta usar,
itera hasta llegar a la conclusión y recuerda la conversación por `thread_id`.

## Qué hay adentro

| Archivo | Qué hace |
|---|---|
| `tools.py` | Las herramientas (`@tool`): consulta a una base técnica, cálculo de prioridad y fecha. |
| `agent.py` | `StateGraph(MessagesState)` + `ToolNode` + arista condicional `tools_condition`. |
| `traza_ejecucion.json` | Traza del razonamiento (se genera al correr `agent.py`). |
| `tests/` | Pruebas de las herramientas y del armado del grafo. |

## Cómo correrlo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # completar OPENAI_API_KEY
python agent.py             # ejecuta la demo y escribe traza_ejecucion.json
```

La consulta de la demo se puede cambiar con la variable `CONSULTA`.

## Variables de entorno

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | `openai`, `anthropic` o `gemini` (por defecto `openai`). |
| `OPENAI_API_KEY` | Requerida si el proveedor es OpenAI. |
| `GOOGLE_API_KEY` | Requerida si el proveedor es Gemini (free tier, sin tarjeta). |
| `CONSULTA` | Opcional: consulta a ejecutar en la demo. |

## Ejemplo de traza (razonamiento multi-paso)

```json
[
  {"tipo": "HumanMessage", "contenido": "Necesito el estado de la orden ORD-1001 y ..."},
  {"tipo": "AIMessage", "herramientas": [
      {"nombre": "consultar_base_de_datos", "argumentos": {"identificador": "ORD-1001"}},
      {"nombre": "calcular_prioridad", "argumentos": {"criticidad": "alta", "dias_demora": 3}},
      {"nombre": "fecha_actual", "argumentos": {}}
  ]},
  {"tipo": "ToolMessage", "contenido": "{\"identificador\": \"ORD-1001\", \"estado\": ...}"},
  {"tipo": "ToolMessage", "contenido": "Prioridad P1 — escalar ahora, SLA 4 horas."},
  {"tipo": "ToolMessage", "contenido": "2026-09-12T18:00:00+00:00"},
  {"tipo": "AIMessage", "contenido": "La orden ORD-1001 está en preparación y ..."}
]
```

El ciclo es: `modelo → herramientas → modelo → …`, hasta que el modelo responde sin pedir
más herramientas.

## Decisiones de diseño

- **Estado**: el grafo hereda de `MessagesState`; el estado se **acumula** (los mensajes se
  suman mediante el reducer), no se reemplaza.
- **Arista condicional**: `tools_condition` mira si el último mensaje trae `tool_calls`. Si
  trae, va al nodo de herramientas; si no, termina. Es lo que hace que el agente itere solo,
  sin `if/else` manuales.
- **Persistencia**: `AsyncSqliteSaver` guarda el estado por `thread_id` en
  `checkpoints.sqlite`, por lo que el agente recuerda la sesión. Pasando el mismo
  `thread_id` en una segunda consulta, se acuerda de lo anterior.
- **Herramientas**: los docstrings son deliberadamente descriptivos, porque el modelo decide
  cuál usar **solo** en base a esa descripción.
- **Recursion limit**: se define un techo de 10 pasos para evitar bucles infinitos y costos
  inesperados.

## Tests

```bash
pytest -q
```

Se prueban las herramientas (son funciones puras) y que el grafo se arme con los nodos y
la arista condicional que pide la consigna.
