# Databricks notebook source
# MAGIC %pip install langchain langchain-groq langgraph
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os
import mlflow
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope",
    key="grok_key"
)

llm = ChatGroq(model="llama-3.3-70b-versatile")

# ── Tools ─────────────────────────────────────────────────────────
@tool
def add_numbers(a: float, b: float) -> float:
    """Adds two numbers together. Use this when you need to add numbers."""
    return a + b

@tool
def calculate_percentage(value: float, percentage: float) -> float:
    """Calculates what percentage of a value is.
    For example calculate_percentage(500, 10) returns 50.0
    which is 10% of 500."""
    return (value * percentage) / 100

# ── Agent ─────────────────────────────────────────────────────────
agent = create_react_agent(
    model=llm,
    tools=[add_numbers, calculate_percentage],
    state_modifier = "You are a data enginner working on databricks"
)

# ── MLflow ────────────────────────────────────────────────────────
mlflow.set_experiment("/Users/dattada.vijay@gmail.com/agent-learning")
mlflow.langchain.autolog()

# ── Run the agent ─────────────────────────────────────────────────
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is 10% of the sum of 500 and 300?"}
    ]
})


# ── Print every message in the conversation ───────────────────────
for message in result["messages"]:
    print(f"{message.type.upper()}: {message.content}")
    print()





