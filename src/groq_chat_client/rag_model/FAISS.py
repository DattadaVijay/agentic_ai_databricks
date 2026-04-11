# Databricks notebook source
# MAGIC %pip install langchain langchain-groq langchain-community sentence-transformers faiss-cpu langchain-text-splitters pypdf --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
doc_path = dbutils.widgets.get("doc_file_path")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
    scope="agents_scope",
    key="grok_key"
)
os.environ["HF_TOKEN"] = dbutils.secrets.get(
    scope="agents_scope",
    key="hugging_face_key"
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

loader = PyPDFLoader(doc_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

vector_store  = FAISS.from_documents(
    documents=chunks,
    embedding=embedding
)

vector_store.save_local("/Workspace/Users/dattada.vijay@gmail.com/RAG/faiss_index")

# From now on I can use FAISS.load_local to directly load the index

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

def format_docs(docs):
    result = ""
    for doc in docs:
        result += doc.page_content
        result += "\n\n"
    return result.strip() 

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
        "context": retriever | format_docs,
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
