from get_models import chat_model_generator
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever


def myretriever_mmr(chromaDB):
    return chromaDB.as_retriever(search_kwargs={"k": 5, "lambda_mult": 0.2}, search_type="mmr")

def myretriever_similarity(chromaDB):
    return chromaDB.as_retriever(search_kwargs={"k": 5}, search_type="similarity")

def myretriever_multiquery(chromaDB):
    llm = chat_model_generator(model_name="gemini-2.0-flash")
    return MultiQueryRetriever.from_llm(retriever=chromaDB.as_retriever(search_kwargs={"k": 5}), llm=llm)

def my_ensemble_retriever(chromaDB, docstore=None):
    retrievers = [
        myretriever_similarity(chromaDB),
        myretriever_mmr(chromaDB),
        myretriever_multiquery(chromaDB),
    ]

    weights = [0.30, 0.40,0.30] 
    return EnsembleRetriever(
        retrievers=retrievers,
        weights=weights
    )