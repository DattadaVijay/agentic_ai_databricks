# Databricks notebook source
# MAGIC %pip install langchain langchain-groq langgraph
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import mlflow
import pandas as pd

input_example = pd.DataFrame({
    "questions": [
        "What is the job ID of [dev dattada_vijay] agentic_ai_databricks_job?"
    ]
})

mlflow.set_experiment("/Users/dattada.vijay@gmail.com/databricks_governance_agent")

with mlflow.start_run(run_name="databricks_governance_agent"):

    mlflow.log_param("model",    "llama-3.3-70b-versatile")
    mlflow.log_param("endpoint", "groq")
    mlflow.log_param("tools",    "get_job_id, get_job_creator")

    mlflow.pyfunc.log_model(
        name="databricks_governance_agent",
        python_model="/Workspace/Users/dattada.vijay@gmail.com/.bundle/agentic_ai_databricks/dev/files/src/groq_chat_client/data_steward/agent_3.py",
        pip_requirements=[
            "langchain",
            "langchain-groq",
            "langgraph"
        ],
        input_example=input_example
    )

    run_id = mlflow.active_run().info.run_id
    print(f"✅ Model logged")
    print(f"Run ID:    {run_id}")
    print(f"Model URI: runs:/{run_id}/databricks_governance_agent")


dbutils.jobs.taskValues.set(key = "run_id", value = run_id)