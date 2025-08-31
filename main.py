import streamlit as st
from get_retreiver import my_ensemble_retriever
from get_document_files_uploader import document_uploader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from get_vector_store_generator import vector_store
from get_splitter import Splitter_chunker
from get_models import chat_model_generator
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime
from langchain.memory import ConversationBufferMemory 


def save_chat_history(chat_history, filename="texthistory.txt"):
    """Save chat history to a text file with timestamps"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Session: {timestamp}\n")
        f.write(f"{'='*50}\n")
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                f.write(f"[{timestamp}] User: {msg.content}\n")
            elif isinstance(msg, AIMessage):
                f.write(f"[{timestamp}] Assistant: {msg.content}\n")
        f.write("\n")


st.title("PDF Analyzer")

# Initialize session variables
if "chromaDB" not in st.session_state:
    st.session_state.chromaDB = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:  
    st.session_state.memory = ConversationBufferMemory(memory_key="history", input_key="user_input", return_messages=False)

# Button to reset chat and memory
if st.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.success("Chat history and memory cleared!")

# Upload PDF documents
documents = document_uploader(st)
parser = StrOutputParser()

# Button to create vector store from uploaded PDFs
if st.button("Start"):
    if documents:
        for doc in documents:
            doc.page_content = doc.page_content.replace("\n", " ")

        chunk_splitter = Splitter_chunker(splitter="RecursiveCharacterTextSplitter")
        docs_split = []
        for doc in documents:
            chunks = chunk_splitter.split_documents([doc])
            docs_split.extend(chunks)

        st.session_state.chromaDB = vector_store(docs_split)
        st.success("Vector store created successfully!")
    else:
        st.warning("Please upload at least one PDF document.")

# Display entire chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content)

# Input box at bottom
if st.session_state.chromaDB:
    user_input = st.chat_input("Ask a question about your PDFs:")

    if user_input:
        #  Add user input to memory
        st.session_state.memory.chat_memory.add_user_message(user_input)

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.chat_message("user").markdown(user_input)

        model = chat_model_generator(model_name="gemini-2.0-flash")
        retriever = my_ensemble_retriever(st.session_state.chromaDB)

        relevant_docs = retriever.invoke(user_input) or []

        if not relevant_docs:
            answer = "Insufficient information in the DB."
        else:
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            #  Fetch conversation history from memory
            history_str = "\n".join(
                [f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
                 for m in st.session_state.memory.chat_memory.messages]
            )

            template = PromptTemplate(template=(
                "You are a research assistant who could analyze the pdf and answer the questions. Answer the question using ONLY the provided context. "
                "Refer to the conversation history for continuity, but do not add extra facts beyond the context.\n\n"

                "=== Chat History ===\n{history}\n\n"
                "=== Context ===\n{context}\n\n"

                "=== Instructions ===\n"
                "1. If the user’s question is about the conversation itself (e.g., listing previous questions, summarizing past interactions, or any meta-question), answer it using **only the chat history** and dont give a reference to that. \n "
                "2. Base your answer strictly on the provided context. If the answer is not in the context, say: "
                "'The provided context does not contain this information.'\n"
                "3. Use a formal, academic tone.\n"
                "4. Include inline citations in the form [1], [2], etc., corresponding to the context excerpts.\n Trim the sentences on the reference if they have irrelevant additional information."
                "5. After the answer, add a 'References' section listing each cited source number and its full excerpt.\n"
                "6. If multiple sources are relevant, cite all in numerical order.\n"
               

                "Question:\n{user_input}\n\n"

                "Example:\n"
                "Question: List all magical spells mentioned in the text along with their effects.\n"
                "Answer: The text mentions Wingardium Leviosa [1], Alohomora [2], and Expelliarmus [3].\n\n"
                "References:\n"
                "[1] Hermione demonstrated Wingardium Leviosa, a charm used to levitate objects by swishing and flicking the wand while pronouncing the incantation.\n\n"
                "[2] Harry whispered Alohomora, a simple unlocking charm, to open the locked door leading to the third-floor corridor.\n\n"
                "[3] With a swift motion, Harry cast Expelliarmus, the disarming charm, causing Draco's wand to fly out of his hand."
            ),
            input_variables=["history", "context", "user_input"])


            chain = template | model | parser
            answer = chain.invoke({
                "context": context,
                "user_input": user_input,
                "history": st.session_state.memory
            })

        # Add AI response to memory
        st.session_state.memory.chat_memory.add_ai_message(answer)

        st.session_state.chat_history.append(AIMessage(content=answer))
        st.chat_message("assistant").markdown(answer)

        save_chat_history(st.session_state.chat_history)