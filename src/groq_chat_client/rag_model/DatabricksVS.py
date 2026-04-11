# Databricks notebook source
# DBTITLE 1,Install Vector Search SDK
# MAGIC %pip install databricks-vectorsearch langchain-community databricks-langchain --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### I am creating my indexe here

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

INDEX_NAME    = "dltvijay.default.candidate_profiles"
ENDPOINT_NAME = "resume_docs"

index = vsc.create_direct_access_index(
    endpoint_name=ENDPOINT_NAME,
    index_name=INDEX_NAME,
    primary_key="id",
    embedding_dimension=1024,
    embedding_vector_column="text_vector",
    schema={
        "id":           "string",
        "page_content": "string",
        "source":       "string",
        "page":         "int",
        "text_vector":  "array<float>"
    }
)

print(f"✅ Index created: {INDEX_NAME}")
print(index.describe())

# COMMAND ----------

loader = PyPDFLoader(
    "/Volumes/digital_twin_dev/airport_ops/raw_governance_files/cavallo_governance_policies.pdf"
)
pages  = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks   = splitter.split_documents(pages)
print(f"Pages:  {len(pages)}")
print(f"Chunks: {len(chunks)}")

# COMMAND ----------

vector_store = DatabricksVectorSearch(
    index_name=INDEX_NAME,
    embedding=embeddings,
    text_column="page_content"
)

vector_store.add_documents(chunks)
print(f"✅ Stored {len(chunks)} chunks in Databricks Vector Search")





# COMMAND ----------

# MAGIC %md
# MAGIC ### Retriever and everything is same