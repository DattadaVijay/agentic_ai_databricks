# Databricks notebook source

# MAGIC %pip install langchain langchain-groq
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os
import mlflow
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope",
    key="grok_key"
)

# COMMAND ----------
# LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# COMMAND ----------
# Tool 1 — Add numbers
@tool
def add_numbers(a: float, b: float) -> float:
    """Adds two numbers together. Use this when you need to add numbers."""
    return a + b

# Tool 2 — Calculate percentage
@tool
def calculate_percentage(value: float, percentage: float) -> float:
    """Calculates what percentage of a value is.
    For example calculate_percentage(500, 10) returns 50.0
    which is 10% of 500."""
    return (value * percentage) / 100

# COMMAND ----------
# Inspect tools
print("--- Tool 1 ---")
print("Name:       ", add_numbers.name)
print("Description:", add_numbers.description)
print("Args:       ", add_numbers.args)

print("\n--- Tool 2 ---")
print("Name:       ", calculate_percentage.name)
print("Description:", calculate_percentage.description)
print("Args:       ", calculate_percentage.args)

# COMMAND ----------
# Call tools manually
result1 = add_numbers.invoke({"a": 10, "b": 25})
print("add_numbers(10, 25)                      =", result1)

result2 = calculate_percentage.invoke({"value": 500000, "percentage": 10})
print("calculate_percentage(500000, 10)         =", result2)