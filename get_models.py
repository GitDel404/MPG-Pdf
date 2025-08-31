from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio

load_dotenv()

def embedding_model_generator(model_name):
    # Ensure an event loop exists (needed in Python 3.13 threads like Streamlit)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    return GoogleGenerativeAIEmbeddings(model=model_name)

def chat_model_generator(model_name):
    # Same loop check in case you use async gRPC here too
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    return ChatGoogleGenerativeAI(model=model_name)