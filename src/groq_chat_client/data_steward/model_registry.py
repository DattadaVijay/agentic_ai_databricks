# Databricks notebook source
import mlflow
import os

os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope",
    key="grok_key"
)

run_id = dbutils.widgets.get("run_id")
registered = mlflow.register_model(
    model_uri=f"runs:/{run_id}/databricks_governance_agent",
    name="databricks-governance-agent"
)

# COMMAND ----------

print(f"Name:    {registered.name}")
print(f"Version: {registered.version}")