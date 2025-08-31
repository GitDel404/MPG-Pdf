from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional, Any


def Splitter_chunker(embedding_model: Optional[Any] = None, splitter = 'RecursiveCharacterTextSplitter'):
    if splitter == "RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,  chunk_overlap=200)
    elif splitter == "SemanticChunker":
        text_splitter = SemanticChunker(embeddings=embedding_model, breakpoint_threshold_type='standard_deviation', breakpoint_threshold_amount=1)
    else:
        text_splitter= None
    
    return text_splitter