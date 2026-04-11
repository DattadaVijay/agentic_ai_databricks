# Databricks notebook source
# MAGIC %pip install langchain langchain-community sentence-transformers faiss-cpu chromadb pinecone-client langchain-groq pypdf --quiet 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma, Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
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

# COMMAND ----------

import logging
import warnings
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model = "llama-3.3-70b-versatile")

# COMMAND ----------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

loader = PyPDFLoader("/Workspace/Users/dattada.vijay@gmail.com/RAG/resume.pdf")
pages = loader.load()

print(f"\nPage 1 content (first 300 chars):")
print(pages[0].page_content[:300])
print(f"\nPage 1 metadata:")
print(pages[0].metadata)

# COMMAND ----------

chunks = splitter.split_documents(pages)


vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="/Workspace/Users/dattada.vijay@gmail.com/RAG/chroma_db",
    collection_name="resume"
)

retreiver = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

response = retreiver.invoke("Total years of work experience?")



# COMMAND ----------

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful HR assistant.
Answer questions about the candidate from their resume.
Answer ONLY based on the context provided below.
If the answer is not in the context say 'I do not have that information.'

Context:
{context}"""),
    ("user", "{question}")
])

# COMMAND ----------

rag_chain = (
    {
        "context": retreiver | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

# COMMAND ----------

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