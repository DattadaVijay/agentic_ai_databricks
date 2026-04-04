# Databricks notebook source
# Calling the Groq using the standard way.
%pip install groq
dbutils.library.restartPython()
# COMMAND ----------
import os
from groq import Groq

os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope", 
    key="grok_key"
)

client = Groq()

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful data engineering assistant."},
        {"role": "user",   "content": "What is a Delta table in 2 sentences?"}
    ]
)
# COMMAND ----------
print(response.choices[0].message.content)

# COMMAND ----------

# Calling the Groq using the langchain. - What are the benifits?
# Just changing a line works for any agent - example: ChatGroq to ChatOpenAi
# Also here we can chain the messages pass previous outpusts to next using pipe | operator

# COMMAND ----------

%pip install langchain langchain-groq
dbutils.library.restartPython()

# COMMAND ----------

import os
from langchain_groq import ChatGroq
os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope", 
    key="grok_key"
)

llm = ChatGroq(model = "llama-3.3-70b-versatile")

response = llm.invoke("What is a Delta table in 2 sentences?")

# COMMAND ----------
print(response.content)

# COMMAND ----------
# Now let us learn how we can pass both the system message and the input message

import os
from langchain_core.messages import  SystemMessage, HumanMessage
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope", 
    key="grok_key"
)

llm = ChatGroq(model = "llama-3.3-70b-versatile")


message = [
    SystemMessage("You are a databricks and data engineering specialist who can teach very well"),
    HumanMessage("Explain spark declarative pipeline vs dbt pipeline for databricks in short")
]
response = llm.invoke(message)

print(response.content
)
