# Databricks notebook source
# DBTITLE 1, Session Scoped Installs

# MAGIC %pip install langchain langchain-community sentance-transformers langchain-text-splitters chromadb

# COMMAND ----------

# DBTITLE 1, Understanding how embeddings and vectordb work

from langchain_cumminity.embeddings import HuggingFaceEmbeddings
from langchain_cumminity.vectorstores import Chroma
from langchain.text_splitters import RecursiveCharecterTextSplitter

Docs = ["PII columns including name, email, phone number and national ID must be masked using Databricks column masking functions. This is required under GDPR Article 25.",
    "Table freshness SLA requires all operational tables to be updated before 08:30 every day. Any table not updated by this time is considered a VIOLATION.",
    "Job SLA compliance is measured over a 24 hour window. A job is COMPLIANT if 80 percent or more of its runs succeed. Below 80 percent is NON_COMPLIANT.",
    "The business glossary is the single source of truth for column tagging. All columns must be tagged against a glossary term with similarity score above 0.92.",
    "Data retention policy requires raw data to be retained for 7 years. Processed data for 3 years. Personal data must be deleted after 2 years.",
    "Access to sensitive schemas requires approval from the data governance committee. Service principals must follow the least privilege principle.",]

# DBTITLE 1, Now lets say i want to understand how embeddings and vector db integrate

embeddings = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

splitter = RecursiveCharecterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10
)

chunks = splitter.create_documents(Docs)

vector_stores = Chroma.from_documents(
    documents = chunks,
    embeddings = embeddings,
    collection_names = "Governance_Policies"
)

# COMMAND ----------
print(f"Docs:    {len(Docs)}")
print(f"Chunks:  {len(chunks)}")
print(f"Stored:  {vector_stores._collection.count()}")










