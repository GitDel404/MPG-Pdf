from langchain_community.vectorstores import Chroma
import streamlit as st
from get_models  import embedding_model_generator


def vector_store(documents):
    embedding_model = embedding_model_generator(model_name="models/embedding-001")
    chromadb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name="ChromaDB",
        #persist_directory="Chroma_DB"
    )
    return chromadb