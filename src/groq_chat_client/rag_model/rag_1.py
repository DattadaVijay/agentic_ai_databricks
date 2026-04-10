# Databricks notebook source
# DBTITLE 1,Cell 1
# MAGIC %pip install langchain langchain-community sentence-transformers faiss-cpu chromadb pinecone-client --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 2
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

docs = [
    "PII columns including name, email, phone number and national ID must be masked using Databricks column masking functions. This is required under GDPR Article 25. Table freshness SLA requires all operational tables to be updated before 08:30 every day. Any table not updated by this time is considered a VIOLATION. Job SLA compliance is measured over a 24 hour window. A job is COMPLIANT if 80 percent or more of its runs succeed. Below 80 percent is NON_COMPLIANT. The business glossary is the single source of truth for column tagging. All columns must be tagged against a glossary term with similarity score above 0.92. Data retention policy requires raw data to be retained for 7 years. Processed data for 3 years. Personal data must be deleted after 2 years. Access to sensitive schemas requires approval from the data governance committee. Service principals must follow the least privilege principle."
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap =50
)

chunks = splitter.create_documents(docs)


vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    collection_name="policy_docs"
)

print(f"\n✅ Stored {vector_store._collection.count()} chunks in Chroma")

# COMMAND ----------

query = "Which columns need to be masked?"
results = vector_store.similarity_search(query, k=2)

print(f"Query: {query}\n")
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}")
    print()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Above we are manually controlling search now let us create a retriver that everytime searches with the given config

# COMMAND ----------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

results = retriever.invoke("What is the masking policy for PII?")

print(f"Retrieved {len(results)} chunks:\n")
for i, doc in enumerate(results):
    print(f"Chunk {i+1}: {doc.page_content}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Testing what foramat docs does

# COMMAND ----------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# COMMAND ----------


retriever = retriever.invoke("Which columns need masking?")

print("Before format_docs:")
print(type(retrieved_docs))
print(type(retrieved_docs[0])) 
print()

context_string = format_docs(retrieved_docs)

print("After format_docs:")
print(type(context_string))       
print(context_string)