# Databricks notebook source

# MAGIC %pip install lanchain langchain-groq langgraph
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import mlflow
import pandas as pd

mlflow.set_experiment("Users/dattada.vijay@gmail.com/databricks_governance_agent")

test_input = pd.DataFrame({
    "prompt": [
        "What is the job ID of [dev dattada_vijay] agentic_ai_databricks_job?",
        "Who created the job with ID 96407719029696?"
    ]
})

with mlflow.start_run(name = "databricks_governance_agent"):
    mlflow.log_param("name", "databricks_governance_agent")
    mlflow.log_param("endpoint", "ChatGroq")
    mlflow.log_param("tools", "[get_job_id, get_job_creator]")

    mlflow.pyfunc.log_model(
        name = "databricks_governance_agent",
        python_model = "agent3.py",
        pip_requirements=[
            "langchain",
            "langgraph",
            "langchain-groq"
        ],
        input_example = test_input
    )

run_id = mlflow.active_run().info.run_id
print(f"✅ Model logged")
print(f"Run ID:    {run_id}")
print(f"Model URI: runs:/{run_id}/job_lookup_agent")