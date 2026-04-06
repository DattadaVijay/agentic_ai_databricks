# Databricks notebook source

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

llm = ChatGroq(model = "llama-3.3-70b-versatile")

@tool
def get_job_id(job_name: str) -> str:
    """Gets the Databricks job ID for a given job name.
    Use this when you need to find the job ID of a Databricks job.
    IMPORTANT: Pass the EXACT full job name including any brackets, 
    prefixes or special characters. Do not modify the job name in any way.
    For example '[dev dattada_vijay] my_job' should be passed exactly as is.
    Returns the job ID as a string, or an error message if not found."""
    
    df = spark.table("system.lakeflow.jobs")
    matches = df.filter(df.name == job_name).select("job_id").collect()
    
    if not matches:
        return f"No job found with name '{job_name}'"
    
    return str(matches[0][0])

@tool
def get_job_status(job_id: str) -> str:
    """Gets the latest run status of a Databricks job given its job ID.
    Returns the status of the most recent run or an error if not found."""
    df = spark.table("system.lakeflow.job_runs")
    matches = (df
        .filter(df.job_id == job_id)
        .orderBy(df.start_time.desc())
        .select("job_id", "run_state")
        .limit(1)
        .collect()
    )
    if not matches:
        return f"No runs found for job ID '{job_id}'"
    return str(matches[0]["run_state"])


agent = create_react_agent(
    model=llm,
    tools=[get_job_id, get_job_status],
    prompt="""You are a Databricks governance expert with access to job information.
You can look up job IDs and job run statuses.
Always use the exact job name as provided by the user.
When you find information, summarise it clearly.
Remember previous answers in the conversation and refer to them when relevant."""
)

job_name = dbutils.widgets.get("job_name")

messages = dbutils.widgets.get("messages")

print(messages)


