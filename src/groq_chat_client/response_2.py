# Databricks notebook source
# Parameterising the prompts
%pip install langchain langchain_groq
dbutils.library.restartPython()

# COMMAND ----------
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate # This helps me parameterise the prompts

os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope", 
    key="grok_key"
)

llm = ChatGroq(model = "llama-3.3-70b-versatile")

prompt = ChatPromptTemplate([
("system", "You are a {platform} {field} specialist and a teacher who use proper tables or bullets to teach in the most simple way"),
("user", "Teach me {topic} in {length} lines")
])

filled_prompt = prompt.format_messages(
    platform = dbutils.widgets.get("platform"),
    field = dbutils.widgets.get("field"),
    topic = dbutils.widgets.get("topic"),
    length = dbutils.widgets.get("length")
)

print(filled_prompt)

response = llm.invoke(filled_prompt)

print(response.content)

# COMMAND ----------

# Now let us learn the outputparser

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

clean = parser.invoke(response)

print(type(clean))
print(clean)
