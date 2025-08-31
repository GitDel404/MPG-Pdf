# PDF Analyzer

A Streamlit-based PDF analysis application that uses Google's Gemini models for document processing and intelligent question answering with conversation memory.

## Features
- **Document Processing**: Upload and process multiple PDF documents
- **Semantic Search**: Advanced retrieval using ensemble methods
- **Interactive Chat**: Natural language Q&A with document context
- **Conversation Memory**: Persistent chat history with context awareness
- **Citation System**: Referenced responses with source excerpts
- **Session Management**: Reset functionality and persistent storage

## Setup Steps

### Prerequisites
- Python 3.8 or higher
- Google AI API key
- Internet connection for model inference

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MPG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_ai_api_key_here
   ```
   
   **Note**: Obtain your API key from [Google AI Studio](https://aistudio.google.com/apikey)

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Upload PDF documents using the file uploader
   - Click "Start" to process documents and create the vector store
   - Begin asking questions in the chat interface

## Models and Architecture

### Language Models

#### Chat Model
- **Model**: `gemini-2.0-flash`
- **Provider**: Google Generative AI
- **Purpose**: Question answering and text generation
- **Context Window**: 1M tokens
- **Parameters**: ~2.6B
- **Performance**: Optimized for speed and efficiency

#### Embedding Model
- **Model**: `models/embedding-001`
- **Provider**: Google Generative AI
- **Purpose**: Document vectorization for semantic search
- **Dimensions**: 768
- **Parameters**: ~137M
- **Performance**: Optimized for retrieval tasks

### Document Processing Strategy

#### Chunking Strategy
- **Primary Splitter**: `RecursiveCharacterTextSplitter`
- **Chunk Size**: 1500 characters
- **Overlap**: 200 characters
- **Alternative**: Semantic chunking available (requires embedding model)

#### Vector Database
- **Database**: ChromaDB
- **Storage**: Persistent storage with SQLite backend
- **Indexing**: Automatic vector indexing

### Retrieval System

The application uses an **Ensemble Retriever** combining three strategies ensures high relevance and robust query coverage:

1. **Similarity Search** (30% weight)
   - Standard vector similarity search
   - Retrieves top 5 most similar documents
   - Ensures the most semantically relevant documents are retrieved.
   Limitation: Can sometimes return very similar documents (lack of diversity) and miss alternate phrasings of the query.

2. **MMR (Maximum Marginal Relevance)** (40% weight)
   - Balances relevance and diversity
   - Lambda multiplier: 0.2
   - Retrieves top 5 documents

3. **Multi-Query Retrieval** (30% weight)
   - Generates multiple queries from user input
   - Captures different ways of expressing the same intent.
   - Improves retrieval coverage
   - Uses LLM for query generation


## Conversation History Management

### Memory System
- **Type**: `ConversationBufferMemory`
- **Storage**: Session state with persistent file backup
- **Key Features**:
  - Maintains full conversation context
  - Supports conversation continuity
  - Handles both user and AI messages

### History Persistence
- **File Storage**: `texthistory.txt`
- **Format**: Timestamped sessions with clear message separation
- **Structure**:
  ```
  ==================================================
  Session: 2024-01-15 14:30:25
  ==================================================
  [2024-01-15 14:30:25] User: What is the main topic?
  [2024-01-15 14:30:26] Assistant: Based on the context...
  ```

### Context Integration
- **Chat History**: Included in prompt template for continuity
- **Memory Key**: "history" for conversation context
- **Input Key**: "user_input" for current questions
- **Reset Functionality**: Clear chat history and memory with button

## Technical Stack

- **Frontend**: Streamlit (web interface)
- **LLM Framework**: LangChain (orchestration)
- **Models**: Google Generative AI (Gemini)
- **Vector Database**: ChromaDB
- **Document Processing**: PyPDF
- **Environment Management**: python-dotenv
- **Memory**: LangChain ConversationBufferMemory

## Usage Workflow

1. **Document Upload**: Use the file uploader to select PDF documents
2. **Processing**: Click "Start" to chunk and vectorize documents
3. **Questioning**: Ask questions in natural language
4. **Responses**: Receive answers with citations and references
5. **History**: View complete conversation history
6. **Reset**: Clear conversation when needed

## Known Limitations

### Model Limitations
- **Context Window**: Limited by Gemini's 1M token context
- **API Rate Limits**: Subject to Google AI API quotas
- **Model Availability**: Dependent on Google AI service status

### Document Processing
- **PDF Quality**: Poorly scanned or corrupted PDFs may not process correctly
- **Language Support**: Optimized for English text
- **File Size**: Large documents may take longer to process
- **Format Support**: Currently only supports PDF format

### Retrieval Limitations
- **Chunk Size**: Fixed 1500-character chunks may split important context
- **Semantic Accuracy**: Retrieval quality depends on embedding model performance
- **Query Complexity**: Very specific or technical queries may have limited results

### Application Limitations
- **Session Persistence**: Chat history is session-based (clears on page refresh)
- **Concurrent Users**: Single-user application (no multi-user support)
- **Offline Mode**: Requires internet connection for model inference
- **Memory Usage**: Large documents may consume significant memory

### Performance Considerations
- **Processing Time**: Large documents require longer processing time
- **Response Latency**: Dependent on API response times
- **Memory Footprint**: Vector store and embeddings stored in memory

## Troubleshooting

### Common Issues
1. **API Key Error**: Ensure `.env` file contains valid `GOOGLE_API_KEY`
2. **Import Errors**: Verify all dependencies are installed via `requirements.txt`
3. **Memory Issues**: Restart application for large document sets
4. **Empty Responses**: Check document quality and try rephrasing questions

### Performance Tips
- Use smaller PDF files for faster processing
- Reset chat history periodically to free memory
- Ensure stable internet connection for optimal performance
