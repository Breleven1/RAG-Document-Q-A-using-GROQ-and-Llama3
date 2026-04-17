import streamlit as st
import os
import time

from dotenv import load_dotenv
load_dotenv()

# ENV
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# LLM
from langchain_groq import ChatGroq
llm = ChatGroq(model_name="Llama3-8b-8192")

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Updated import
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector DB + Loader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Prompt
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the context.

<context>
{context}
</context>

Question: {input}
""")

# Create embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        loader = PyPDFDirectoryLoader("research_papers")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        documents = splitter.split_documents(docs[:50])

        vectorstore = FAISS.from_documents(documents, embeddings)

        st.session_state.vectors = vectorstore
        st.session_state.documents = documents


# UI
st.title("RAG Document Q&A With Groq + Llama3")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

# -------- LCEL PIPELINE --------
from langchain_core.runnables import RunnablePassthrough

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please click 'Document Embedding' first.")
    else:
        retriever = st.session_state.vectors.as_retriever()

        # LCEL chain (modern way)
        chain = (
            {
                "context": retriever,
                "input": RunnablePassthrough()
            }
            | prompt
            | llm
        )

        start = time.process_time()
        response = chain.invoke(user_prompt)
        end = time.process_time()

        st.write(response.content)
        st.write(f"Response time: {end - start:.2f} sec")

        # Show retrieved docs
        with st.expander("Document similarity Search"):
            docs = retriever.get_relevant_documents(user_prompt)
            for i, doc in enumerate(docs):
                st.write(doc.page_content)
                st.write('------------------------')