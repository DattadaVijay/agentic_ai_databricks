# Databricks notebook source

# COMMAND ----------
# MAGIC %pip install langchain langchain-groq langgraph
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope",
    key="grok_key"
)

llm = ChatGroq(model="llama-3.3-70b-versatile")

# COMMAND ----------
@tool
def get_job_id(job_name: str) -> str:
    """Gets the Databricks job ID for a given job name.
    Use this when you need to find the job ID of a Databricks job.
    Returns the job ID as a string, or an error message if not found."""
    
    df = spark.table("system.lakeflow.jobs")
    matches = df.filter(df.name.contains(job_name)).select("name", "job_id").collect()
    
    if not matches:
        return f"No job found containing '{job_name}'"
    
    return str(matches[0][0])

# COMMAND ----------
agent = create_react_agent(
    model=llm,
    tools=[get_job_id],
    prompt="You are a Databricks governance expert"
)

job_name = dbutils.widgets.get("job_name")

messages = [{"role": "user", "content": f"What is the job ID of {job_name}?"}]

response = agent.invoke({"messages": messages})

# COMMAND ----------
# Print full ReAct loop
for message in response["messages"]:
    print(f"{message.type.upper()}: {message.content}")
    print()
