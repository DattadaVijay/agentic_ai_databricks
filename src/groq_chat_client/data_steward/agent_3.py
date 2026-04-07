# Databricks notebook source
# MAGIC %pip install lanchain langchain-groq langgraph
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
import os
import mlflow
import pandas as pd

class DataGovernanceAgent(mlflow.pyfunc.PythonModel):

    def load_context(self, context):

        os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
        scope="agents_scope",
        key="grok_key")

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
        def get_job_creator(job_id: str) -> str:
            """Gets the creator username of a Databricks job given its job ID.
            Use this when you need to find who created a job.
            If only a job name is given, first use get_job_id to get the job ID,
            then pass that job ID to this tool.
            Returns the creator username or an error message if not found."""
            df = spark.table("system.lakeflow.jobs")
            matches = df.filter(df.job_id == job_id).select("creator_user_name").collect()
            if not matches:
                return f"No job found with job ID '{job_id}'"
            return str(matches[0][0])
        
        self.agent = create_agent(
            model = llm,
            tools = [get_job_id, get_job_creator],
            system_prompt="""You are a Databricks governance expert with access to job information. You can look up job IDs and job run statuses. Always use the exact job name as provided by the user. When you find information, summarise it clearly. Remember previous answers in the conversation and refer to them when relevant."""
        )

    def predict(self, context, model_input):
        result = []
        for question in model_input["questions"]:
            message = [{"role": "user", "content": question}]
            response = self.agent.invoke({
                "messages": message
            })
            result.append(response["messages"][-1].content)

mlflow.models.set_model(DataGovernanceAgent())
        


# test_input = pd.DataFrame({
#     "prompt": [
#         "What is the job ID of [dev dattada_vijay] agentic_ai_databricks_job?",
#         "Who created the job with ID 96407719029696?"
#     ]
# })