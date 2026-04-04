# Databricks notebook source

# COMMAND ----------
%pip install groq
dbutils.library.restartPython()
# COMMAND ----------
import os
from groq import Groq

os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope", 
    key="groq_api_key"
)

client = Groq()

response = client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages = [
        {
            {"role": "system", "content": "You are a helpful data engineering assistant."},
            {"role": "user",   "content": "What is a Delta table in 2 sentences?"}
        }
    ]
)
# COMMAND ----------
print(response.choices[0].message.content)



