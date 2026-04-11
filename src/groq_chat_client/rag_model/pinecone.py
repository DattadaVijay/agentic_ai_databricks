# Databricks notebook source
# MAGIC %pip install pinecone-client langchain-pinecone langchain-community sentence-transformers langchain-groq --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from pinecone import Pinecone, ServerlessSpec

os.environ["PINECONE_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope",
    key="pinecone_key"
)

pc = Pinecone()

print("Existing indexes:")
for index in pc.list_indexes():
    print(f"  - {index.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating my Index in Pinecone

# COMMAND ----------

INDEX_NAME = "candidate-profiles"
existing_index_names = [index.name for index in pc.list_indexes()]

if INDEX_NAME not in existing_index_names:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index {INDEX_NAME} already exists, skipping creation")



# COMMAND ----------

# MAGIC %md
# MAGIC ### describe_index can be used to print attributes of any index

# COMMAND ----------

print(f"✅ Index '{INDEX_NAME}' created and ready")
print(f"Dimension: {pc.describe_index(INDEX_NAME).dimension}")
print(f"Metric:    {pc.describe_index(INDEX_NAME).metric}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now i have my PC client active and the index created. I am going to add embeddings to it

# COMMAND ----------

# MAGIC %pip install pypdf

# COMMAND ----------

from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq  import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

os.environ["HF_TOKEN"] = dbutils.secrets.get(
    scope="agents_scope",
    key="hugging_face_key"
)
os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope",
    key="grok_key"
)

#splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

#loader = PyPDFLoader("/Workspace/Users/dattada.vijay@gmail.com/RAG/resume.pdf")

#doc = loader.load()

#chunks = splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings)

# COMMAND ----------

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)

def format_docs(docs):
    context = " "
    for doc in docs:
        context += doc.page_content + "\n" + "\n"
    return context


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Answer the following question based on the context below.
         You should use the information in the context to answer the question.
        If the information is not there, say that you don't know.
        
        Context: {Context}
        """),

        ("user", "{question}")

    ]
)

llm = ChatGroq(model = "llama-3.3-70b-versatile")

parser = StrOutputParser()


# COMMAND ----------

rag_chain = (
    {
        "Context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)


questions = [
    "What is the name of the candidate ?",
    "What is the candidate's email address?",
    "What is the candidate's phone number?",
    "What is the candidate's total years of work experience?",
    "What is the candidate's total years of experience in the field of data engineering?"
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {rag_chain.invoke(q)}")
    print()
