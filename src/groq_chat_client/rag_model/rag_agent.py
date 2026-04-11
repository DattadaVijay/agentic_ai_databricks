# Databricks notebook source
# MAGIC %pip install langchain-pinecone pinecone-client langchain-groq langchain langchain-community pypdf sentence-transformers --quiet
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import mlflow
from langchain_pinecone import PineconeVectorStore
import pandas as pd
import os

# COMMAND ----------

class rag_agent(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        os.environ["GROQ_API_KEY"] = dbutils.secrets.get(
        scope="agents_scope",
        key="grok_key"
        )
        os.environ["HF_TOKEN"] = dbutils.secrets.get(
            scope="agents_scope",
            key="hugging_face_key"
        )

        os.environ["PINECONE_API_KEY"] = dbutils.secrets.get(
        scope="agents_scope",
        key="pinecone_key"
        )


        loader = PyPDFLoader("/Workspace/Users/dattada.vijay@gmail.com/RAG/resume.pdf")

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en")
        llm = ChatGroq(model="llama-3.3-70b-versatile")

        pc = Pinecone()

        # pc.create_index(
        #     name = "doc-rag",
        #     dimension = 1024,
        #     metric = "cosine",
        #     spec = ServerlessSpec(
        #         cloud="aws",
        #         region="us-east-1"
        # ))
            
        # vector_store = PineconeVectorStore.from_documents(
        #     index_name= "doc-rag",
        #     embedding= embeddings,
        #     documents = chunks
        # )

        vector_store = PineconeVectorStore(
            index_name= "doc-rag",
            embedding= embeddings
        )

        retriever = vector_store.as_retriever(
            search_kwargs = {
                "k": 3
            }
        )

        def format_docs(docs):
            return "\n".join([f"{doc.page_content}" for doc in docs]).strip()

        @tool
        def search_resume(query: str) -> str:
            """Searches candidate resume documents to answer HR questions.
            Use this when asked about a candidate's experience, skills, education,
            job titles, certifications, contact details or work history.
            Returns relevant resume excerpts with page numbers."""
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant information found in the resume."
            return format_docs(docs)

        self.agent = create_agent(
            model = llm,
            tools = [search_resume],
            system_prompt="""You are a helpful assistant that answers questions about a candidate's resume.
            You have access to a search_resume tool that searches the candidate's resume for relevant information.
            Use the search_resume tool to answer questions about the candidate's experience, skills, education,
            job titles, certifications, contact details or work history.
            If the question is not about the candidate's resume, just answer the question without using the tool.""",
        )

    def predict(self, context, model_input: pd.DataFrame) -> list:
        results = []
        for question in model_input["question"]:
            response = self.agent.invoke({
                "messages": [{"role": "user", 
                            "content": question}]
            })
            results.append(response["messages"][-1].content)
        return results


# COMMAND ----------

test_input = pd.DataFrame({
    "question": [         
        "What is the candidate's contact information?",
        "What is the candidate's education?",
        "What is the candidate's work history?",
        "What is the candidate's experience?",
        "What is the candidate's skills?",
        "Did candidate work on databricks?"
    ]
})

agent_model = rag_agent()
agent_model.load_context(None)   

results = agent_model.predict(None, test_input)

for q, a in zip(test_input["question"], results):
    print(f"Q: {q}")
    print(f"A: {a}")
    print()

# COMMAND ----------

